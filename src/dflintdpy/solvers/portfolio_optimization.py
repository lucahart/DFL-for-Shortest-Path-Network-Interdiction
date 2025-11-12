"""
Bilevel Pricing Optimization Problem

Outer (Leader): max_p  p^T y*(p)
Inner (Follower): y*(p) = argmax_y (c - p)^T y
                          s.t. y^T Σ y ≤ γ
                               1^T y ≤ 1
                               y ≥ 0

This is a Stackelberg pricing game where:
- Seller sets prices p
- Buyer responds with optimal portfolio y
- Seller maximizes revenue p^T y
"""

from copy import deepcopy
from xml.parsers.expat import model
import numpy as np
from typing import Tuple, Optional, Dict
import warnings

from pyepo.model.grb import optGrbModel

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


class PortfolioOptimization(optGrbModel):
    """
    Solver for portfolio optimization problems.
    """
    
    def __init__(
            self, 
            c: np.ndarray, 
            Sigma: np.ndarray, 
            gamma: float,
            p_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        ):
        """
        Args:
            c: Cost/return vector (length n)
            p: Initial price vector (length n)
            Sigma: Covariance matrix (n x n), must be PSD
            gamma: Risk tolerance parameter
            p_bounds: (p_min, p_max) bounds for prices. If None, uses (0, c)
        """
        self.c = np.array(c)
        self.Sigma = np.array(Sigma)
        self.gamma = gamma
        self.n = len(c)
        
        # Validate inputs
        assert len(c) == self.n
        assert Sigma.shape == (self.n, self.n)
        assert np.allclose(Sigma, Sigma.T), "Sigma must be symmetric"
        assert gamma > 0
        
        # Check PSD
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals >= -1e-10), "Sigma must be positive semidefinite"

        # Initialize parent class
        super().__init__()
    
    def __deepcopy__(self, memo):
        """
        Create a deep copy of the shortestPathGrb instance.
        
        Parameters:
        -----------
        memo : dict
            A dictionary to keep track of already copied objects.

        Returns:
        --------
        ShortestPathGrb
            A new instance of ShortestPathGrb with the same attributes.
        """
        
        # Create a new instance and copy the graph
        new_instance = PortfolioOptimization(
            self.c.copy(),
            self.Sigma.copy(),
            self.gamma
        )
        return new_instance

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        sol, obj = super().solve()

        if self._model.status==2:
            sol = self.x.x
            sol[sol < 1e-4] = 0.0
        else:
            raise Exception("Optimal Solution not found")   
        
        return sol, obj
    
    def _getModel(self):
        """
        A method to build Gurobi model
        Returns:
            tuple: optimization model and variables
        """

        # Create a new model
        model = gp.Model("portfolio_optimization")

        # Set feasibility tolerance (default: 1e-6)
        # Controls how strictly constraints must be satisfied
        model.Params.FeasibilityTol = 1e-9

        # Create variables: y is a vector of n non-negative variables
        y = model.addMVar(self.n, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

        # Set objective: maximize c^T y
        model.setObjective(self.c @ y, GRB.MAXIMIZE)

        # Add constraints
        # 1. Quadratic constraint: y^T * Sigma * y <= gamma
        model.addConstr(y @ self.Sigma @ y <= self.gamma, name="risk_constraint")

        # 2. Sum constraint: sum(y) <= 1
        model.addConstr(y.sum() <= 1, name="budget_constraint")

        # 3. Non-negativity is already handled by lb=0.0 in addMVar

        return model, y

