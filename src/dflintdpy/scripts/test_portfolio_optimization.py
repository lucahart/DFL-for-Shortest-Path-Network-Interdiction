from dflintdpy.solvers.portfolio_optimization import PortfolioOptimization
from dflintdpy.solvers.bilevel_pricing import BilevelPricingProblem

import numpy as np
import pandas as pd
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import pytest


class TestPortfolioOptimization:
    """Test suite comparing CVXPY and Gurobi implementations of Markowitz portfolio optimization"""
    
    @pytest.fixture
    def simple_portfolio(self):
        """Create a simple 3-asset portfolio for testing"""
        n = 3
        c_pp = np.array([0.10, 0.15, 0.08])  # Expected returns (c - p)
        Sigma = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])
        gamma = 0.05  # Risk tolerance
        
        return {
            'n': n,
            'c_pp': c_pp,
            'Sigma': Sigma,
            'gamma': gamma
        }
    
    @pytest.fixture
    def larger_portfolio(self):
        """Create a larger random portfolio"""
        np.random.seed(42)
        n = 10
        c_pp = np.random.uniform(0.05, 0.20, n)
        
        # Generate a positive semi-definite covariance matrix
        A = np.random.randn(n, n)
        Sigma = A.T @ A / n
        gamma = 0.10
        
        return {
            'n': n,
            'c_pp': c_pp,
            'Sigma': Sigma,
            'gamma': gamma
        }
    
    def solve_cvxpy(self, n, c_pp, Sigma, gamma):
        """Solve using CVXPY"""
        y = cp.Variable(n)
        objective = cp.Maximize(c_pp @ y)
        constraints = [
            cp.quad_form(y, Sigma) <= gamma,
            cp.sum(y) <= 1,
            y >= 0
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return {
            'status': problem.status,
            'objective': problem.value,
            'solution': y.value
        }
    
    def solve_gurobi(self, n, c_pp, Sigma, gamma):
        """Solve using Gurobi"""
        opt_model = PortfolioOptimization(c_pp, Sigma, gamma)
        sol, obj = opt_model.solve()

        return {
            'status': opt_model._model.Status,
            'objective': obj,
            'solution': np.array(sol)
        }
    
    def test_simple_portfolio_optimal_solution(self, simple_portfolio):
        """Test that both solvers find the optimal solution for a simple portfolio"""
        cvxpy_result = self.solve_cvxpy(**simple_portfolio)
        gurobi_result = self.solve_gurobi(**simple_portfolio)
        
        # Check both found optimal solutions
        assert cvxpy_result['status'] == 'optimal'
        assert gurobi_result['status'] == GRB.OPTIMAL
        
        # Check objectives are close
        np.testing.assert_allclose(
            cvxpy_result['objective'], 
            gurobi_result['objective'], 
            rtol=1e-4,
            err_msg="Objective values don't match"
        )
        
        # Check solutions are close
        np.testing.assert_allclose(
            cvxpy_result['solution'], 
            gurobi_result['solution'], 
            atol=1e-4, # Computing the absolute tolerance
            rtol=1,
            err_msg="Solution vectors don't match"
        )
    
    def test_larger_portfolio_optimal_solution(self, larger_portfolio):
        """Test that both solvers agree on a larger random portfolio"""
        cvxpy_result = self.solve_cvxpy(**larger_portfolio)
        gurobi_result = self.solve_gurobi(**larger_portfolio)
        
        assert cvxpy_result['status'] == 'optimal'
        assert gurobi_result['status'] == GRB.OPTIMAL
        
        np.testing.assert_allclose(
            cvxpy_result['objective'], 
            gurobi_result['objective'], 
            rtol=1e-4
        )
        
        np.testing.assert_allclose(
            cvxpy_result['solution'], 
            gurobi_result['solution'], 
            atol=1e-3, # Computeting the absolute tolerance
            rtol=1
        )
    
    def test_constraints_satisfied_gurobi(self, simple_portfolio):
        """Verify Gurobi solution satisfies all constraints"""
        result = self.solve_gurobi(**simple_portfolio)
        
        y = result['solution']
        Sigma = simple_portfolio['Sigma']
        gamma = simple_portfolio['gamma']
        
        # Check risk constraint
        risk = y @ Sigma @ y
        assert risk <= gamma + 1e-6, f"Risk constraint violated: {risk} > {gamma}"
        
        # Check budget constraint
        assert np.sum(y) <= 1 + 1e-6, f"Budget constraint violated: {np.sum(y)} > 1"
        
        # Check non-negativity
        assert np.all(y >= -1e-6), "Non-negativity constraint violated"
    
    def test_constraints_satisfied_cvxpy(self, simple_portfolio):
        """Verify CVXPY solution satisfies all constraints"""
        result = self.solve_cvxpy(**simple_portfolio)
        
        y = result['solution']
        Sigma = simple_portfolio['Sigma']
        gamma = simple_portfolio['gamma']
        
        # Check risk constraint
        risk = y @ Sigma @ y
        assert risk <= gamma + 1e-6, f"Risk constraint violated: {risk} > {gamma}"
        
        # Check budget constraint
        assert np.sum(y) <= 1 + 1e-6, f"Budget constraint violated: {np.sum(y)} > 1"
        
        # Check non-negativity
        assert np.all(y >= -1e-6), "Non-negativity constraint violated"
    
    def test_zero_returns(self):
        """Test edge case where all returns are zero"""
        n = 3
        c_pp = np.zeros(n)
        Sigma = np.eye(n) * 0.01
        gamma = 0.05
        
        cvxpy_result = self.solve_cvxpy(n, c_pp, Sigma, gamma)
        gurobi_result = self.solve_gurobi(n, c_pp, Sigma, gamma)
        
        # Both should find optimal solution with objective = 0
        assert cvxpy_result['status'] == 'optimal'
        assert gurobi_result['status'] == GRB.OPTIMAL
        assert abs(cvxpy_result['objective']) < 1e-6
        assert abs(gurobi_result['objective']) < 1e-6
    
    def test_tight_risk_constraint(self):
        """Test case where risk constraint is very tight"""
        n = 3
        c_pp = np.array([0.10, 0.15, 0.08])
        Sigma = np.eye(n) * 0.1  # High variance
        gamma = 0.01  # Very tight risk constraint
        
        cvxpy_result = self.solve_cvxpy(n, c_pp, Sigma, gamma)
        gurobi_result = self.solve_gurobi(n, c_pp, Sigma, gamma)
        
        assert cvxpy_result['status'] == 'optimal'
        assert gurobi_result['status'] == GRB.OPTIMAL
        
        # Solutions should be similar despite tight constraint
        np.testing.assert_allclose(
            cvxpy_result['objective'], 
            gurobi_result['objective'], 
            rtol=1e-4
        )
    
    def test_single_asset_optimal(self):
        """Test case where single asset should be optimal"""
        n = 3
        c_pp = np.array([0.20, 0.05, 0.05])  # First asset has much higher return
        Sigma = np.eye(n) * 0.01  # Low, equal variance
        gamma = 0.05
        
        cvxpy_result = self.solve_cvxpy(n, c_pp, Sigma, gamma)
        gurobi_result = self.solve_gurobi(n, c_pp, Sigma, gamma)
        
        # Both should primarily invest in first asset
        assert cvxpy_result['solution'][0] > 0.8, "CVXPY should favor first asset"
        assert gurobi_result['solution'][0] > 0.8, "Gurobi should favor first asset"
        
        np.testing.assert_allclose(
            cvxpy_result['solution'], 
            gurobi_result['solution'], 
            rtol=1,
            atol=1e-4
        )
    
    def test_objective_value_calculation(self, simple_portfolio):
        """Verify objective value matches manual calculation"""
        result = self.solve_gurobi(**simple_portfolio)
        
        y = result['solution']
        c_pp = simple_portfolio['c_pp']
        
        manual_objective = c_pp @ y
        
        np.testing.assert_allclose(
            result['objective'], 
            manual_objective, 
            rtol=1e-6,
            err_msg="Objective value doesn't match manual calculation"
        )
    
    def test_performance_comparison(self, larger_portfolio):
        """Compare solve times (informational, not a strict test)"""
        import time
        
        # Time CVXPY
        start = time.time()
        cvxpy_result = self.solve_cvxpy(**larger_portfolio)
        cvxpy_time = time.time() - start
        
        # Time Gurobi
        start = time.time()
        gurobi_result = self.solve_gurobi(**larger_portfolio)
        gurobi_time = time.time() - start
        
        print(f"\nCVXPY solve time: {cvxpy_time:.4f}s")
        print(f"Gurobi solve time: {gurobi_time:.4f}s")
        print(f"Speedup: {cvxpy_time/gurobi_time:.2f}x")
        
        # Just check both solved successfully
        assert cvxpy_result['status'] == 'optimal'
        assert gurobi_result['status'] == GRB.OPTIMAL


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])


