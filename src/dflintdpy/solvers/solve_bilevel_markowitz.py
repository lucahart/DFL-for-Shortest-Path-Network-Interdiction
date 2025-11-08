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

import numpy as np
from typing import Tuple, Optional, Dict
import warnings

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


class BilevelPricingProblem:
    """
    Solver for bilevel pricing optimization problems.
    """
    
    def __init__(self, c: np.ndarray, Sigma: np.ndarray, gamma: float,
                 p_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Args:
            c: Cost/return vector (length n)
            Sigma: Covariance matrix (n x n), must be PSD
            gamma: Risk tolerance parameter
            p_bounds: (p_min, p_max) bounds for prices. If None, uses (0, c)
        """
        self.c = np.array(c)
        self.Sigma = np.array(Sigma)
        self.gamma = gamma
        self.n = len(c)
        
        # Default price bounds: 0 ≤ p ≤ c (prices can't exceed costs)
        if p_bounds is None:
            self.p_min = np.zeros(self.n)
            self.p_max = self.c.copy()
        else:
            self.p_min, self.p_max = p_bounds
        
        # Validate inputs
        assert len(c) == self.n
        assert Sigma.shape == (self.n, self.n)
        assert np.allclose(Sigma, Sigma.T), "Sigma must be symmetric"
        assert gamma > 0
        
        # Check PSD
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals >= -1e-10), "Sigma must be positive semidefinite"
    
    def solve_inner_problem(self, p: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve inner problem for given price vector p.
        
        Returns:
            y_opt: Optimal portfolio
            obj: Inner objective value
        """
        if not HAS_CVXPY:
            raise ImportError("CVXPY required for inner problem. Install: pip install cvxpy")
        
        y = cp.Variable(self.n)
        
        # Inner objective: max (c - p)^T y
        objective = cp.Maximize((self.c - p) @ y)
        
        constraints = [
            cp.quad_form(y, self.Sigma) <= self.gamma,
            cp.sum(y) <= 1,
            y >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Inner problem failed: {problem.status}")
        
        return y.value, problem.value
    
    def evaluate_revenue(self, p: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate revenue p^T y*(p) for given prices.
        
        Returns:
            revenue: p^T y*(p)
            y_opt: Optimal buyer response
        """
        y_opt, _ = self.solve_inner_problem(p)
        revenue = p @ y_opt
        return revenue, y_opt
    
    def solve_with_gurobi_miqcp(self, M: float = 1000, 
                                time_limit: float = 300) -> Optional[Dict]:
        """
        Solve using Gurobi with Mixed-Integer Quadratically Constrained Programming.
        This is the most rigorous approach.
        
        Args:
            M: Big-M constant for complementarity conditions
            time_limit: Maximum solve time in seconds
            
        Returns:
            Dictionary with solution details
        """
        if not HAS_GUROBI:
            print("Gurobi not available. Install: pip install gurobipy")
            return None
        
        model = gp.Model("bilevel_pricing")
        model.setParam('OutputFlag', 1)
        model.setParam('TimeLimit', time_limit)
        model.setParam('NonConvex', 2)  # Allow bilinear terms
        model.setParam('MIPGap', 1e-4)
        
        # Decision variables
        p = model.addMVar(self.n, lb=self.p_min, ub=self.p_max, name="p")
        y = model.addMVar(self.n, lb=0, name="y")
        
        # Dual variables for inner problem
        lambda1 = model.addVar(lb=0, name="lambda1")
        lambda2 = model.addVar(lb=0, name="lambda2")
        mu = model.addMVar(self.n, lb=0, name="mu")
        
        # Binary variables for complementarity
        z1 = model.addVar(vtype=GRB.BINARY, name="z1")  # For risk constraint
        z2 = model.addVar(vtype=GRB.BINARY, name="z2")  # For budget constraint
        z_y = model.addMVar(self.n, vtype=GRB.BINARY, name="z_y")  # For y >= 0
        
        # Auxiliary variables for bilinear terms in objective
        # Objective: max p^T y = max Σ pᵢ yᵢ
        # We need to linearize this bilinear term
        py = model.addMVar(self.n, lb=-GRB.INFINITY, name="py")
        
        # McCormick relaxation for py[i] = p[i] * y[i]
        for i in range(self.n):
            p_l, p_u = self.p_min[i], self.p_max[i]
            y_l, y_u = 0, 1  # y is in [0,1] due to budget constraint
            
            model.addConstr(py[i] >= p_l * y[i] + p[i] * y_l - p_l * y_l)
            model.addConstr(py[i] >= p_u * y[i] + p[i] * y_u - p_u * y_u)
            model.addConstr(py[i] <= p_l * y[i] + p[i] * y_u - p_l * y_u)
            model.addConstr(py[i] <= p_u * y[i] + p[i] * y_l - p_u * y_l)
        
        # Outer objective: max p^T y
        model.setObjective(py.sum(), GRB.MAXIMIZE)
        
        # Inner problem primal feasibility
        model.addConstr(y @ self.Sigma @ y <= self.gamma, name="risk")
        model.addConstr(y.sum() <= 1, name="budget")
        
        # KKT stationarity for inner problem
        # ∇_y L = -(c - p) + 2λ₁Σy + λ₂·1 - μ = 0
        for i in range(self.n):
            grad_i = -(self.c[i] - p[i]) + 2 * lambda1 * gp.quicksum(
                self.Sigma[i, j] * y[j] for j in range(self.n)
            ) + lambda2 - mu[i]
            model.addConstr(grad_i == 0, name=f"kkt_stat_{i}")
        
        # Complementarity slackness using Big-M method
        
        # 1. λ₁ · (y^T Σ y - γ) = 0
        slack1 = y @ self.Sigma @ y - self.gamma
        model.addConstr(slack1 >= -M * z1, name="comp1a")
        model.addConstr(lambda1 <= M * (1 - z1), name="comp1b")
        
        # 2. λ₂ · (1^T y - 1) = 0
        slack2 = y.sum() - 1
        model.addConstr(slack2 >= -M * z2, name="comp2a")
        model.addConstr(lambda2 <= M * (1 - z2), name="comp2b")
        
        # 3. μᵢ · yᵢ = 0 for all i
        for i in range(self.n):
            model.addConstr(y[i] <= M * z_y[i], name=f"comp3a_{i}")
            model.addConstr(mu[i] <= M * (1 - z_y[i]), name=f"comp3b_{i}")
        
        # Solve
        print("Solving bilevel pricing problem with Gurobi MIQCP...")
        print(f"Problem size: n={self.n}, binary vars={2 + self.n}")
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            p_opt = np.array([p[i].X for i in range(self.n)])
            y_opt = np.array([y[i].X for i in range(self.n)])
            revenue = model.objVal
            
            # Verify solution
            y_verify, inner_obj = self.solve_inner_problem(p_opt)
            
            return {
                'p_opt': p_opt,
                'y_opt': y_opt,
                'revenue': revenue,
                'inner_obj': inner_obj,
                'y_verify': y_verify,
                'verification_gap': np.linalg.norm(y_opt - y_verify),
                'success': True,
                'mip_gap': model.MIPGap if hasattr(model, 'MIPGap') else 0,
                'solve_time': model.Runtime,
                'status': 'optimal'
            }
        elif model.status == GRB.TIME_LIMIT:
            try:
                p_opt = np.array([p[i].X for i in range(self.n)])
                y_opt = np.array([y[i].X for i in range(self.n)])
                return {
                    'p_opt': p_opt,
                    'y_opt': y_opt,
                    'revenue': model.objVal,
                    'success': False,
                    'status': 'time_limit',
                    'mip_gap': model.MIPGap,
                    'solve_time': model.Runtime
                }
            except:
                pass
        
        print(f"Gurobi optimization status: {model.status}")
        return None
    
    def solve_with_cvxpy_penalty(self, max_iterations: int = 20,
                                 penalty_increase: float = 10) -> Optional[Dict]:
        """
        Solve using penalty method to approximate complementarity.
        Less rigorous but doesn't require MIQCP.
        
        Args:
            max_iterations: Maximum penalty iterations
            penalty_increase: Factor to increase penalty each iteration
            
        Returns:
            Dictionary with solution details
        """
        if not HAS_CVXPY:
            print("CVXPY not available.")
            return None
        
        print("Solving with CVXPY penalty method...")
        
        # Variables
        p = cp.Variable(self.n)
        y = cp.Variable(self.n)
        lambda1 = cp.Variable(nonneg=True)
        lambda2 = cp.Variable(nonneg=True)
        mu = cp.Variable(self.n, nonneg=True)
        
        # Start with small penalty
        rho = 1.0
        best_revenue = -np.inf
        best_result = None
        
        for iteration in range(max_iterations):
            # KKT stationarity
            grad = -(self.c - p) + 2 * lambda1 * (self.Sigma @ y) + lambda2 * np.ones(self.n) - mu
            
            # Penalty terms for complementarity (approximate)
            slack1 = cp.quad_form(y, self.Sigma) - self.gamma
            slack2 = cp.sum(y) - 1
            comp_penalty = rho * (
                cp.sum_squares(lambda1 * slack1) +
                cp.sum_squares(lambda2 * slack2) +
                cp.sum_squares(cp.multiply(mu, y))
            )
            
            # Objective with penalty
            objective = cp.Maximize(p @ y - comp_penalty)
            
            constraints = [
                # Price bounds
                p >= self.p_min,
                p <= self.p_max,
                # Primal feasibility
                cp.quad_form(y, self.Sigma) <= self.gamma,
                cp.sum(y) <= 1,
                y >= 0,
                # Stationarity
                grad == 0
            ]
            
            problem = cp.Problem(objective, constraints)
            
            try:
                problem.solve(solver=cp.CLARABEL)
                
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    revenue = p.value @ y.value
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_result = {
                            'p_opt': p.value,
                            'y_opt': y.value,
                            'revenue': revenue,
                            'iteration': iteration,
                            'penalty': rho
                        }
                    
                    print(f"Iteration {iteration}: revenue={revenue:.4f}, penalty={rho:.2e}")
                else:
                    print(f"Iteration {iteration}: {problem.status}")
            except Exception as e:
                print(f"Iteration {iteration} failed: {e}")
            
            rho *= penalty_increase
        
        if best_result is not None:
            # Verify solution
            y_verify, inner_obj = self.solve_inner_problem(best_result['p_opt'])
            best_result['y_verify'] = y_verify
            best_result['inner_obj'] = inner_obj
            best_result['verification_gap'] = np.linalg.norm(best_result['y_opt'] - y_verify)
            best_result['success'] = True
            best_result['status'] = 'penalty_converged'
        
        return best_result
    
    def solve_with_grid_search(self, n_points: int = 10) -> Dict:
        """
        Solve using grid search over price space.
        Simple but only practical for small n.
        
        Args:
            n_points: Number of grid points per dimension
            
        Returns:
            Dictionary with solution details
        """
        print(f"Grid search with {n_points}^{self.n} = {n_points**self.n} evaluations")
        
        if self.n > 3 and n_points > 10:
            warnings.warn(f"Grid search with n={self.n} and {n_points} points is expensive!")
        
        best_revenue = -np.inf
        best_p = None
        best_y = None
        
        # Create grid for each price
        grids = [np.linspace(self.p_min[i], self.p_max[i], n_points) 
                 for i in range(self.n)]
        
        import itertools
        count = 0
        total = n_points ** self.n
        
        for p_vals in itertools.product(*grids):
            count += 1
            p = np.array(p_vals)
            
            try:
                revenue, y = self.evaluate_revenue(p)
                
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_p = p
                    best_y = y
                
                if count % max(1, total // 20) == 0:
                    print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
            except:
                continue
        
        return {
            'p_opt': best_p,
            'y_opt': best_y,
            'revenue': best_revenue,
            'success': best_p is not None,
            'status': 'grid_search_complete',
            'evaluations': count
        }

