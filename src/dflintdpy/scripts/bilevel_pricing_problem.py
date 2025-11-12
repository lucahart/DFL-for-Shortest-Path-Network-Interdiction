from dflintdpy.data.adverse.adverse_data_generator import AdvDataGenerator
from dflintdpy.data.adverse.adverse_loader import AdvLoader
from dflintdpy.data.config import HP
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
from dflintdpy.data.adverse.adverse_dataset import AdvDataset, generate_opt_dataset
from dflintdpy.predictors.linear_regression import LinearRegression
from dflintdpy.solvers.fast_solvers.fast_pricing_solver import FastBilevelPricingSolver
from dflintdpy.scripts.read_synthetic_data import read_synthetic_data
from dflintdpy.scripts.setup import gen_train_data
from dflintdpy.scripts.evaluate_models import evaluate_multiple_samples, summarize_results, print_summary

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

def simple_dfp_example(N = 1000, noise=1, deg=4, batch_size=32):
    """Simple example of solving a bilevel pricing problem with dfl data."""

    cfg = HP()

    ################## DataReading
    x_train, y_train, x_valid, y_valid, x_test, y_test, cov, gamma = \
        read_synthetic_data(N, noise, deg)
    
    x_train = x_train[:200,:]
    y_train = y_train[:200,:]
    # x_valid = x_valid[:100,:]
    # y_valid = y_valid[:100,:]
    # x_test = x_test[:100,:]
    # y_test = y_test[:100,:]

    ################## ModelCreation
    opt_model = PortfolioOptimization(y_train[0,:], Sigma=cov, gamma=gamma)

    ################## Generate Interdiction Data
    adverse_generator = AdvDataGenerator(
        cfg, 
        opt_model, 
        budget=0.3 * y_train[0,:].sum(),
        normalization_constant=1.0,
        num_scenarios=1,
        adverse_problem="BPPO",
    )
    adfl_x_train, adfl_y_train, p_train = adverse_generator.generate(
        x_train, 
        y_train
    )
    # adfl_x_val, adfl_y_val, p_val = adverse_generator.generate(
    #     x_valid,
    #     y_valid
    # )

    # Create data sets
    # dataset_train_adverse = AdvDataset(
    #     opt_model,
    #     x_train.reshape(x_train.shape[0], 1, x_train.shape[1]),
    #     y_train.reshape(y_train.shape[0], 1, y_train.shape[1]),
    #     np.ones_like(y_train)
    # )
    dataset_train_adverse = AdvDataset(opt_model, adfl_x_train, 
                                       adfl_y_train, p_train)
    # val_dataset = AdvDataset(opt_model, adfl_x_val, adfl_y_val, p_val)

    # Create data loaders for training and validation
    loader_train_adverse = AdvLoader(
        dataset_train_adverse,
        batch_size=cfg.get("batch_size"),
        seed=cfg.get("loader_seed"),
        shuffle=True,
    )
    # loader_val_adverse = AdvLoader(
    #     val_dataset,
    #     batch_size=cfg.get("batch_size"),
    #     seed=cfg.get("loader_seed"),
    #     shuffle=False,
    # )

    ################## DataLoader
    dataset_train = generate_opt_dataset(opt_model, x_train, y_train)
    dataset_valid = generate_opt_dataset(opt_model, x_valid, y_valid)
    dataset_test = generate_opt_dataset(opt_model, x_test, y_test)

    loader_train = AdvLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = AdvLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    loader_test = AdvLoader(dataset_test, batch_size=batch_size, shuffle=False)

    ################## Train simple model
    # Instantiate linear regression model
    pred_model_dfl = LinearRegression(
        num_feat=x_train.shape[1], 
        num_edges=opt_model.num_cost
    )
    pred_model_pfl = LinearRegression(
        num_feat=x_train.shape[1], 
        num_edges=opt_model.num_cost
    )
    pred_model_adfl = LinearRegression(
        num_feat=x_train.shape[1], 
        num_edges=opt_model.num_cost
    )

    # Init SPO+ loss
    dfl_loss = pyepo.func.SPOPlus(opt_model, processes=1)
    pfl_loss = torch.nn.MSELoss()
    adfl_loss = pyepo.func.SPOPlus(opt_model, processes=1)

    # Init optimizer
    optimizer = optim.Adam(pred_model_dfl.parameters(), lr=1e-2)
    optimizer_pfl = optim.Adam(pred_model_pfl.parameters(), lr=1e-2)
    optimizer_adfl = optim.Adam(pred_model_adfl.parameters(), lr=1e-2)

    # Set the number of epochs for training
    epochs = 2

    # Create a trainer instance
    dfl_trainer = DFLTrainer(
        pred_model=pred_model_dfl, 
        opt_model=opt_model,
        optimizer=optimizer, 
        loss_fn=dfl_loss
    )
    pfl_trainer = PFLTrainer(
        pred_model=pred_model_pfl,
        opt_model=opt_model,
        optimizer=optimizer_pfl,
        loss_fn=pfl_loss
    )
    adfl_trainer = DFLTrainer(
        pred_model=pred_model_adfl, 
        opt_model=opt_model,
        optimizer=optimizer_adfl, 
        loss_fn=adfl_loss
    )

    # Train the models
    dfl_train_loss_log, dfl_train_regret_log, dfl_val_loss_log, dfl_val_regret_log = \
        dfl_trainer.fit(loader_train, loader_valid, epochs=epochs)
    pfl_train_loss_log, pfl_train_regret_log, pfl_val_loss_log, pfl_val_regret_log = \
        pfl_trainer.fit(loader_train, loader_valid, epochs=epochs)
    adfl_train_loss_log, adfl_train_regret_log, adfl_val_loss_log, adfl_val_regret_log = \
        adfl_trainer.fit(loader_train_adverse, loader_valid, epochs=epochs)
    
    # Visualize learning curves
    DFLTrainer.vis_learning_curve(
        dfl_trainer,
        dfl_train_loss_log,
        dfl_train_regret_log,
        dfl_val_loss_log,
        dfl_val_regret_log,
        file_name="figures/dfl_learning_curve"
    )
    PFLTrainer.vis_learning_curve(
        pfl_trainer,
        pfl_train_loss_log,
        pfl_train_regret_log,
        pfl_val_loss_log,
        pfl_val_regret_log,
        file_name="figures/pfl_learning_curve"
    )
    DFLTrainer.vis_learning_curve(
        adfl_trainer,
        adfl_train_loss_log,
        adfl_train_regret_log,
        adfl_val_loss_log,
        adfl_val_regret_log,
        file_name="figures/adfl_learning_curve"
    )

    # Print final regrets

    print("DFL: Final regret on validation set:   ", dfl_val_regret_log[-1])
    print("PFL: Final regret on validation set:   ", pfl_val_regret_log[-1])
    print("A-DFL: Final regret on validation set: ", adfl_val_regret_log[-1])

    # Evaluate on test set
    loader_test.adverse_mode()
    _, dfl_test_regret = dfl_trainer.evaluate(loader_test)
    print("DFL: Final regret on test set: ", dfl_test_regret)
    loader_test.normal_mode()
    _, pfl_test_regret = pfl_trainer.evaluate(loader_test)
    print("PFL: Final regret on test set: ", pfl_test_regret)
    loader_test.adverse_mode()
    _, adfl_test_regret = adfl_trainer.evaluate(loader_test)
    print("A-DFL: Final regret on test set: ", adfl_test_regret)


    # Run on 50 test samples
    all_results = evaluate_multiple_samples(
        num_samples=50,
        x_test=x_test,
        y_test=y_test,
        pred_model_dfl=pred_model_dfl,
        pred_model_pfl=pred_model_pfl,
        pred_model_adfl=pred_model_adfl,
        opt_model=opt_model,
        cov=cov,
        gamma=gamma,
        fast_only=True,
    )

    # Get summary statistics
    summary = summarize_results(all_results)
    print_summary(summary)
    # idx = 0
    # c = y_test[idx,:]
    # x = x_test[idx,:]
    # dfl_pred = pred_model_dfl(torch.tensor(x)).detach().numpy()
    # pfl_pred = pred_model_pfl(torch.tensor(x)).detach().numpy()
    # adfl_pred = pred_model_adfl(torch.tensor(x)).detach().numpy()
    # print("\n" + "=" * 80)
    # print("BILEVEL PRICING OPTIMIZATION ON TEST SAMPLE")
    # print("=" * 80)
    # opt_model.setObj(dfl_pred)
    # sol_dfl, _ = opt_model.solve()
    # opt_model.setObj(pfl_pred)
    # sol_pfl, _ = opt_model.solve()
    # opt_model.setObj(adfl_pred)
    # sol_adfl, _ = opt_model.solve()
    # print("\nProblem Parameters:")
    # print(f"Cost vector c: {c}")
    # # print(f"Buyer's solution: {sol_dfl}")
    # print(f"DFL: Buyer's objective value:   {sol_dfl @ c:.4f}")
    # print(f"PFL: Buyer's objective value:   {sol_pfl @ c:.4f}")
    # print(f"A-DFL: Buyer's objective value: {sol_adfl @ c:.4f}")
    # print(f"Risk tolerance γ: {gamma}")
    # print(f"Problem size: n = {c.shape[0]}")
    # print(f"Budget: {c.sum()*0.3}")

    # # Create problem
    # problem = BilevelPricingProblem(c, cov, gamma, budget=c.sum()*0.3)
    # fast_solver = FastBilevelPricingSolver(c, cov, gamma, budget=c.sum()*0.3)
    # result_gurobi = problem.solve_with_gurobi_miqcp(M=100, time_limit=60)
    # result_fast = fast_solver.solve(n_starts=5)
    # p = result_gurobi['p_opt']
    # p_fast = result_fast['p_opt']
    # opt_model.setObj(dfl_pred-p)
    # sol_pp_dfl, _ = opt_model.solve()
    # opt_model.setObj(pfl_pred-p)
    # sol_pp_pfl, _ = opt_model.solve()
    # opt_model.setObj(adfl_pred-p)
    # sol_pp_adfl, _ = opt_model.solve()
    # opt_model.setObj(dfl_pred-p_fast)
    # sol_ppf_dfl, _ = opt_model.solve()
    # opt_model.setObj(pfl_pred-p_fast)
    # sol_ppf_pfl, _ = opt_model.solve()
    # opt_model.setObj(adfl_pred-p_fast)
    # sol_ppf_adfl, _ = opt_model.solve()
    # if result_gurobi is not None and result_gurobi['success']:
    #     print("\n" + "=" * 80)
    #     print("PLAYER PAYOFFS AFTER PRICING")
    #     print("=" * 80)
    #     print(f"\nOptimal prices p: {result_gurobi['p_opt']}")
    #     print(f"Buyer response y: {result_gurobi['y_opt']}")
    #     print(f"Seller's revenue: {result_gurobi['revenue']:.4f}")
    #     print(f"Seller's revenue (fast solver): {result_fast['revenue']:.4f}")
    #     print(f"DFL: Buyer's objective value:   {sol_pp_dfl @ (c-p):.4f}")
    #     print(f"PFL: Buyer's objective value:   {sol_pp_pfl @ (c-p):.4f}")
    #     print(f"A-DFL: Buyer's objective value: {sol_pp_adfl @ (c-p):.4f}")
    #     print(f"DFL: Buyer's objective value:   {sol_ppf_dfl @ (c-p_fast):.4f}")
    #     print(f"PFL: Buyer's objective value:   {sol_ppf_pfl @ (c-p_fast):.4f}")
    #     print(f"A-DFL: Buyer's objective value: {sol_ppf_adfl @ (c-p_fast):.4f}")
    #     print(f"True values:        {c.mean():.4f}, {c.std():.4f}")
    #     print(f"DFL: Predictions:   {dfl_pred.mean():.4f}, {dfl_pred.std():.4f}")
    #     print(f"PFL: Predictions:   {pfl_pred.mean():.4f}, {pfl_pred.std():.4f}")
    #     print(f"A-DFL: Predictions: {adfl_pred.mean():.4f}, {adfl_pred.std():.4f}")
    #     print(f"Verification gap: {result_gurobi['verification_gap']:.2e}")
    #     print(f"MIP gap: {result_gurobi['mip_gap']:.2e}")
    #     print(f"Solve time: {result_gurobi['solve_time']:.2f}s")
    
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
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    simple_dfp_example()
    # example_usage()