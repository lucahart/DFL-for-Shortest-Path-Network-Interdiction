from dflintdpy.data.adverse.adverse_loader import AdvLoader
from dflintdpy.utils.dfl_trainer import DFLTrainer
from dflintdpy.utils.pfl_trainer import PFLTrainer
import numpy as np
import pandas as pd
import pyepo
import torch
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings

from dflintdpy.solvers.portfolio_optimization import PortfolioOptimization
from dflintdpy.solvers.bilevel_pricing import BilevelPricingProblem
from dflintdpy.data.adverse.adverse_dataset import generate_opt_dataset
from dflintdpy.predictors.linear_regression import LinearRegression
from dflintdpy.solvers.fast_solvers.fast_pricing_solver import FastBilevelPricingSolver
from dflintdpy.scripts.read_synthetic_data import read_synthetic_data
from dflintdpy.scripts.setup import gen_train_data

# Try to import optional dependencies
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    warnings.warn("CVXPY not available. Some methods will not work.")

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    warnings.warn("Gurobi not available. MIQCP method will not work.")

def simple_dfp_example(N = 1000, noise=1, deg=16, batch_size=32):
    """Simple example of solving a bilevel pricing problem with dfl data."""

    ################## DataReading
    x_train, y_train, x_valid, y_valid, x_test, y_test, cov, gamma = \
        read_synthetic_data(N, noise, deg)

    ################## ModelCreation
    opt_model = PortfolioOptimization(y_train[0,:], Sigma=cov, gamma=gamma)

    ################## DataLoader
    dataset_train = generate_opt_dataset(opt_model, x_train, y_train)
    dataset_valid = generate_opt_dataset(opt_model, x_valid, y_valid)
    dataset_test = generate_opt_dataset(opt_model, x_test, y_test)

    loader_train = AdvLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = AdvLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    loader_test = AdvLoader(dataset_test, batch_size=batch_size, shuffle=False)

    ################## Train simple model
    # Instantiate linear regression model
    pred_model = LinearRegression(
        num_feat=x_train.shape[1], 
        num_edges=opt_model.num_cost
    )
    pred_model_pfl = LinearRegression(
        num_feat=x_train.shape[1], 
        num_edges=opt_model.num_cost
    )

    # Init SPO+ loss
    spop = pyepo.func.SPOPlus(opt_model, processes=1)
    pfl_loss = torch.nn.MSELoss()

    # Init optimizer
    optimizer = optim.Adam(pred_model.parameters(), lr=1e-2)
    optimizer_pfl = optim.Adam(pred_model_pfl.parameters(), lr=1e-2)

    # Set the number of epochs for training
    epochs = 5

    # Create a trainer instance
    trainer = DFLTrainer(
        pred_model=pred_model, 
        opt_model=opt_model,
        optimizer=optimizer, 
        loss_fn=spop
    )
    pfl_trainer = PFLTrainer(
        pred_model=pred_model_pfl,
        opt_model=opt_model,
        optimizer=optimizer_pfl,
        loss_fn=pfl_loss
    )

    # Train the model
    train_loss_log, train_regret_log, valid_loss_log, valid_regret_log = \
        trainer.fit(loader_train, loader_valid, epochs=epochs)

    # Train the model
    train_loss_log, train_regret_log, val_loss_log, val_regret_log = \
        pfl_trainer.fit(loader_train, loader_valid, epochs=epochs)
    
    # Visualize learning curves
    DFLTrainer.vis_learning_curve(
        trainer,
        train_loss_log,
        train_regret_log,
        valid_loss_log,
        valid_regret_log,
        file_name="figures/dfl_learning_curve"
    )

    # Plot the learning curve
    PFLTrainer.vis_learning_curve(
        pfl_trainer,
        train_loss_log,
        train_regret_log,
        val_loss_log,
        val_regret_log,
        file_name="figures/pfl_learning_curve"
    )

    # Print final regrets

    print("DFL: Final regret on validation set: ", valid_regret_log[-1])
    print("PFL: Final regret on validation set: ", val_regret_log[-1])

    # Evaluate on test set
    loader_test.adverse_mode()
    test_loss, test_regret = trainer.evaluate(loader_test)
    print("DFL: Final regret on test set: ", test_regret)
    loader_test.normal_mode()
    test_loss_pfl, test_regret_pfl = pfl_trainer.evaluate(loader_test)
    print("PFL: Final regret on test set: ", test_regret_pfl)


    # Bilevel pricing
    idx = 1
    c = y_test[idx,:]
    x = x_test[idx,:]
    dfl_pred = pred_model(torch.tensor(x)).detach().numpy()
    pfl_pred = pred_model_pfl(torch.tensor(x)).detach().numpy()
    print("\n" + "=" * 80)
    print("BILEVEL PRICING OPTIMIZATION ON TEST SAMPLE")
    print("=" * 80)
    opt_model.setObj(dfl_pred)
    sol_dfl, _ = opt_model.solve()
    opt_model.setObj(pfl_pred)
    sol_pfl, _ = opt_model.solve()
    print("\nProblem Parameters:")
    print(f"Cost vector c: {c}")
    # print(f"Buyer's solution: {sol_dfl}")
    print(f"DFL: Buyer's objective value: {sol_dfl @ c:.4f}")
    print(f"PFL: Buyer's objective value: {sol_pfl @ c:.4f}")
    print(f"Risk tolerance γ: {gamma}")
    print(f"Problem size: n = {c.shape[0]}")
    print(f"Budget: {c.sum()*0.3}")

    # Create problem
    problem = BilevelPricingProblem(c, cov, gamma, budget=c.sum()*0.3)        
    result_gurobi = problem.solve_with_gurobi_miqcp(M=100, time_limit=60)
    p = result_gurobi['p_opt']
    opt_model.setObj(dfl_pred-p)
    sol_pp_dfl, _ = opt_model.solve()
    opt_model.setObj(pfl_pred-p)
    sol_pp_pfl, _ = opt_model.solve()
    if result_gurobi is not None and result_gurobi['success']:
        print("\n" + "=" * 80)
        print("PLAYER PAYOFFS AFTER PRICING")
        print("=" * 80)
        print(f"\nOptimal prices p: {result_gurobi['p_opt']}")
        print(f"Buyer response y: {result_gurobi['y_opt']}")
        print(f"Seller's revenue: {result_gurobi['revenue']:.4f}")
        print(f"DFL: Buyer's objective value: {sol_pp_dfl @ (c-p):.4f}")
        print(f"PFL: Buyer's objective value: {sol_pp_pfl @ (c-p):.4f}")
        print(f"Verification gap: {result_gurobi['verification_gap']:.2e}")
        print(f"MIP gap: {result_gurobi['mip_gap']:.2e}")
        print(f"Solve time: {result_gurobi['solve_time']:.2f}s")
    
    # # Test at a specific price point
    # print("\n" + "=" * 80)
    # print("TEST: Buyer response at mid-point prices")
    # print("=" * 80)
    # p_test = (p_min + p_max) / 2
    # revenue_test, y_test = problem.evaluate_revenue(p_test)
    # print(f"Test prices p: {p_test}")
    # print(f"Buyer response y: {y_test}")
    # print(f"Revenue p^T y: {revenue_test:.4f}")
    
    pass



def example_usage():
    """Demonstrate usage"""
    
    print("=" * 80)
    print("BILEVEL PRICING OPTIMIZATION")
    print("=" * 80)
    
    # Problem setup
    np.random.seed(42)
    n = 4  # Keep small for demonstration
    
    # Parameters
    c = np.array([5.0, 6.0, 4.5, 5.5])  # Costs/expected returns
    
    # Create PSD covariance matrix
    A = np.random.randn(n, n)
    Sigma = A.T @ A / n + np.eye(n) * 0.3
    gamma = 0.8
    
    # Price bounds: prices should be below costs for buyer to participate
    p_min = np.zeros(n)
    p_max = c * 0.9  # Max price is 90% of cost
    
    print("\nProblem Parameters:")
    print(f"Cost vector c: {c}")
    print(f"Risk tolerance γ: {gamma}")
    print(f"Price bounds: [{p_min[0]:.2f}, {p_max[0]:.2f}] (same for all)")
    print(f"Problem size: n = {n}")
    
    # Create problem
    problem = BilevelPricingProblem(c, Sigma, gamma, (p_min, p_max))
    
    # Test at a specific price point
    print("\n" + "=" * 80)
    print("TEST: Buyer response at mid-point prices")
    print("=" * 80)
    p_test = (p_min + p_max) / 2
    revenue_test, y_test = problem.evaluate_revenue(p_test)
    print(f"Test prices p: {p_test}")
    print(f"Buyer response y: {y_test}")
    print(f"Revenue p^T y: {revenue_test:.4f}")
    
    # # Grid search
    # print("\n" + "=" * 80)
    # print("METHOD 1: GRID SEARCH")
    # print("=" * 80)
    # result_grid = problem.solve_with_grid_search(n_points=3)
    # if result_grid['success']:
    #     print(f"\nOptimal prices p: {result_grid['p_opt']}")
    #     print(f"Buyer response y: {result_grid['y_opt']}")
    #     print(f"Revenue: {result_grid['revenue']:.4f}")
    
    # Gurobi MIQCP
    if HAS_GUROBI:
        print("\n" + "=" * 80)
        print("METHOD 2: GUROBI MIQCP (Most Rigorous)")
        print("=" * 80)
        result_gurobi = problem.solve_with_gurobi_miqcp(M=100, time_limit=60)
        
        if result_gurobi is not None and result_gurobi['success']:
            print(f"\nOptimal prices p: {result_gurobi['p_opt']}")
            print(f"Buyer response y: {result_gurobi['y_opt']}")
            print(f"Revenue: {result_gurobi['revenue']:.4f}")
            print(f"Verification gap: {result_gurobi['verification_gap']:.2e}")
            print(f"MIP gap: {result_gurobi['mip_gap']:.2e}")
            print(f"Solve time: {result_gurobi['solve_time']:.2f}s")
            
            # # Compare with grid search
            # if result_grid['success']:
            #     print(f"\nComparison:")
            #     print(f"Grid search revenue: {result_grid['revenue']:.4f}")
            #     print(f"Gurobi revenue:      {result_gurobi['revenue']:.4f}")
            #     print(f"Difference:          {abs(result_grid['revenue'] - result_gurobi['revenue']):.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    simple_dfp_example()
    # example_usage()