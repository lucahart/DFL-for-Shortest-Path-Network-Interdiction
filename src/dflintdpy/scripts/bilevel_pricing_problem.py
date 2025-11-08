import numpy as np
from typing import Tuple, Optional, Dict
import warnings

from dflintdpy.solvers.solve_bilevel_markowitz import BilevelPricingProblem

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
    
    # Grid search
    print("\n" + "=" * 80)
    print("METHOD 1: GRID SEARCH")
    print("=" * 80)
    result_grid = problem.solve_with_grid_search(n_points=3)
    if result_grid['success']:
        print(f"\nOptimal prices p: {result_grid['p_opt']}")
        print(f"Buyer response y: {result_grid['y_opt']}")
        print(f"Revenue: {result_grid['revenue']:.4f}")
    
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
            
            # Compare with grid search
            if result_grid['success']:
                print(f"\nComparison:")
                print(f"Grid search revenue: {result_grid['revenue']:.4f}")
                print(f"Gurobi revenue:      {result_gurobi['revenue']:.4f}")
                print(f"Difference:          {abs(result_grid['revenue'] - result_gurobi['revenue']):.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    example_usage()