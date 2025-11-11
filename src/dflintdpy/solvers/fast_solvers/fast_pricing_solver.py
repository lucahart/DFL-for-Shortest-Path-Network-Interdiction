"""
Fast Solvers for Bilevel Pricing with Budget Constraint

max_p  p^T y*(p)
s.t.   1^T p ≤ budget
       p ≥ 0 (implicit)

where y*(p) = argmax_y (c-p)^T y
              s.t. y^T Σ y ≤ γ, 1^T y ≤ 1, y ≥ 0

This module provides FAST methods that sacrifice some rigor for speed.
Perfect when you need good solutions quickly.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, LinearConstraint
from typing import Tuple, Dict, Optional
import warnings

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    warnings.warn("CVXPY not available - inner problem will use scipy")


class FastBilevelPricingSolver:
    """
    Fast solver for bilevel pricing with budget constraint.
    
    Trades mathematical rigor for speed - perfect for practical applications.
    """
    
    def __init__(self, c: np.ndarray, Sigma: np.ndarray, gamma: float, 
                 budget: float, p_max: Optional[np.ndarray] = None):
        """
        Args:
            c: Cost/return vector (length n)
            Sigma: Covariance matrix (n x n), must be PSD
            gamma: Risk tolerance
            budget: Total price budget (1^T p ≤ budget)
            p_max: Maximum price per item (default: c)
        """
        self.c = np.array(c, dtype=float)
        self.Sigma = np.array(Sigma, dtype=float)
        self.gamma = float(gamma)
        self.budget = float(budget)
        self.n = len(c)
        
        if p_max is None:
            self.p_max = self.c.copy()
        else:
            self.p_max = np.array(p_max, dtype=float)
        
        # Validate
        assert Sigma.shape == (self.n, self.n)
        assert np.allclose(Sigma, Sigma.T)
        assert gamma > 0
        assert budget > 0
        
        # Check PSD
        eigvals = np.linalg.eigvalsh(Sigma)
        if not np.all(eigvals >= -1e-10):
            raise ValueError("Sigma must be positive semidefinite")
        
        # Cache for efficiency
        self._y_cache = {}
    
    def solve_inner_problem(self, p: np.ndarray, use_cache: bool = True) -> Tuple[np.ndarray, float]:
        """
        Solve inner problem: max_y (c-p)^T y s.t. constraints
        
        Args:
            p: Price vector
            use_cache: Whether to use cached solutions
            
        Returns:
            y_opt: Optimal portfolio
            obj: Objective value
        """
        # Check cache
        if use_cache:
            p_key = tuple(np.round(p, 6))
            if p_key in self._y_cache:
                return self._y_cache[p_key]
        
        if HAS_CVXPY:
            y = cp.Variable(self.n)
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
            
            result = (y.value, problem.value)
        else:
            # Fallback to scipy
            result = self._solve_inner_scipy(p)
        
        # Cache result
        if use_cache:
            self._y_cache[p_key] = result
        
        return result
    
    def _solve_inner_scipy(self, p: np.ndarray) -> Tuple[np.ndarray, float]:
        """Scipy-based inner solver (fallback)"""
        def objective(y):
            return -np.dot(self.c - p, y)
        
        def risk_constraint(y):
            return self.gamma - y @ self.Sigma @ y
        
        def budget_constraint(y):
            return 1 - np.sum(y)
        
        constraints = [
            {'type': 'ineq', 'fun': risk_constraint},
            {'type': 'ineq', 'fun': budget_constraint}
        ]
        
        bounds = [(0, None) for _ in range(self.n)]
        y0 = np.ones(self.n) / (2 * self.n)
        
        result = minimize(objective, y0, method='SLSQP', 
                         bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-8, 'maxiter': 500})
        
        if not result.success:
            raise ValueError(f"Inner problem failed: {result.message}")
        
        return result.x, -result.fun
    
    def evaluate_revenue(self, p: np.ndarray) -> float:
        """Evaluate revenue p^T y*(p) for given prices"""
        y_opt, _ = self.solve_inner_problem(p)
        return p @ y_opt
    
    def solve_sequential_quadratic(self, p_init: Optional[np.ndarray] = None,
                                   max_iter: int = 100) -> Dict:
        """
        Method 1: Sequential Quadratic Programming (SQP)
        
        Fast gradient-based optimization using L-BFGS-B.
        Uses numerical gradients - no need for explicit derivatives.
        
        SPEED: ⭐⭐⭐⭐⭐ (Very Fast - seconds)
        QUALITY: ⭐⭐⭐⭐ (Good - may find local optima)
        
        Args:
            p_init: Initial price guess
            max_iter: Maximum iterations
            
        Returns:
            Dict with solution
        """
        print("Solving with Sequential Quadratic Programming...")
        
        if p_init is None:
            # Start at uniform prices that satisfy budget
            p_init = np.ones(self.n) * (self.budget / self.n * 0.8)
            p_init = np.minimum(p_init, self.p_max)
        
        def objective(p):
            """Negative revenue (for minimization)"""
            try:
                return -self.evaluate_revenue(p)
            except:
                return 1e10  # Penalty for infeasibility
        
        # Constraints: 1^T p ≤ budget, 0 ≤ p ≤ p_max
        constraints = LinearConstraint(
            np.ones((1, self.n)),  # 1^T p
            lb=0,
            ub=self.budget
        )
        
        bounds = [(0, self.p_max[i]) for i in range(self.n)]
        
        result = minimize(
            objective,
            p_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )
        
        if result.success:
            p_opt = result.x
            y_opt, _ = self.solve_inner_problem(p_opt)
            revenue = -result.fun
            
            return {
                'p_opt': p_opt,
                'y_opt': y_opt,
                'revenue': revenue,
                'success': True,
                'method': 'SQP',
                'iterations': result.nit,
                'time': 'fast'
            }
        else:
            return {'success': False, 'message': result.message}
    
    def solve_projected_gradient(self, p_init: Optional[np.ndarray] = None,
                                 learning_rate: float = 0.1,
                                 max_iter: int = 200,
                                 tolerance: float = 1e-4) -> Dict:
        """
        Method 2: Projected Gradient Ascent
        
        Simple iterative method with projection onto feasible set.
        Very fast, good for warm-starting other methods.
        
        SPEED: ⭐⭐⭐⭐⭐ (Very Fast - seconds)
        QUALITY: ⭐⭐⭐ (Decent - may oscillate)
        
        Args:
            p_init: Initial prices
            learning_rate: Step size
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dict with solution
        """
        print("Solving with Projected Gradient Ascent...")
        
        if p_init is None:
            p = np.ones(self.n) * (self.budget / self.n * 0.8)
            p = np.minimum(p, self.p_max)
        else:
            p = p_init.copy()
        
        def project_onto_feasible(p_raw):
            """Project onto {p: 1^T p ≤ budget, 0 ≤ p ≤ p_max}"""
            # First clip to bounds
            p_proj = np.clip(p_raw, 0, self.p_max)
            
            # Then project onto budget constraint if needed
            if np.sum(p_proj) > self.budget:
                # Binary search for scaling factor
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2
                    if np.sum(np.clip(p_raw * mid, 0, self.p_max)) <= self.budget:
                        lo = mid
                    else:
                        hi = mid
                p_proj = np.clip(p_raw * lo, 0, self.p_max)
            
            return p_proj
        
        best_revenue = -np.inf
        best_p = p.copy()
        
        for iteration in range(max_iter):
            # Compute gradient via finite differences
            y, _ = self.solve_inner_problem(p)
            revenue = p @ y
            
            # Approximate gradient
            grad = np.zeros(self.n)
            eps = 1e-6
            for i in range(self.n):
                p_pert = p.copy()
                p_pert[i] += eps
                if p_pert[i] <= self.p_max[i] and np.sum(p_pert) <= self.budget:
                    y_pert, _ = self.solve_inner_problem(p_pert)
                    revenue_pert = p_pert @ y_pert
                    grad[i] = (revenue_pert - revenue) / eps
                else:
                    grad[i] = 0
            
            # Gradient ascent step
            p_new = p + learning_rate * grad
            
            # Project onto feasible set
            p = project_onto_feasible(p_new)
            
            # Track best
            if revenue > best_revenue:
                best_revenue = revenue
                best_p = p.copy()
            
            # Check convergence
            if iteration > 10 and np.linalg.norm(grad) < tolerance:
                break
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}: revenue = {revenue:.4f}")
        
        y_opt, _ = self.solve_inner_problem(best_p)
        
        return {
            'p_opt': best_p,
            'y_opt': y_opt,
            'revenue': best_revenue,
            'success': True,
            'method': 'Projected Gradient',
            'iterations': iteration + 1
        }
    
    def solve_alternating_optimization(self, p_init: Optional[np.ndarray] = None,
                                      max_iter: int = 50) -> Dict:
        """
        Method 3: Alternating Optimization with Heuristic Updates
        
        Alternates between solving inner problem and updating prices.
        Uses heuristic: increase prices on high-demand items.
        
        SPEED: ⭐⭐⭐⭐⭐ (Very Fast)
        QUALITY: ⭐⭐⭐ (Decent heuristic)
        
        Args:
            p_init: Initial prices
            max_iter: Maximum iterations
            
        Returns:
            Dict with solution
        """
        print("Solving with Alternating Optimization...")
        
        if p_init is None:
            p = np.ones(self.n) * (self.budget / self.n * 0.5)
            p = np.minimum(p, self.p_max)
        else:
            p = p_init.copy()
        
        best_revenue = -np.inf
        best_p = None
        best_y = None
        
        for iteration in range(max_iter):
            # Solve inner problem
            y, _ = self.solve_inner_problem(p)
            revenue = p @ y
            
            # Track best
            if revenue > best_revenue:
                best_revenue = revenue
                best_p = p.copy()
                best_y = y.copy()
            
            # Heuristic: increase prices on items buyer wants more
            # Decrease prices on items buyer doesn't want
            price_adjustment = 0.1 * self.budget / self.n
            
            for i in range(self.n):
                if y[i] > 0.1 / self.n:  # High demand
                    p[i] = min(p[i] + price_adjustment, self.p_max[i])
                elif y[i] < 0.01 / self.n:  # Low demand
                    p[i] = max(p[i] - price_adjustment, 0)
            
            # Ensure budget constraint
            if np.sum(p) > self.budget:
                p = p * (self.budget / np.sum(p))
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: revenue = {revenue:.4f}")
        
        return {
            'p_opt': best_p,
            'y_opt': best_y,
            'revenue': best_revenue,
            'success': True,
            'method': 'Alternating',
            'iterations': max_iter
        }
    
    def solve_multistart(self, n_starts: int = 5, method: str = 'sqp') -> Dict:
        """
        Method 4: Multi-Start Optimization
        
        Run optimization from multiple random starting points.
        Takes the best result.
        
        SPEED: ⭐⭐⭐⭐ (Fast - parallelizable)
        QUALITY: ⭐⭐⭐⭐⭐ (Very Good)
        
        Args:
            n_starts: Number of random starts
            method: Base method to use ('sqp' or 'gradient')
            
        Returns:
            Dict with best solution
        """
        print(f"Solving with Multi-Start ({n_starts} starts)...")
        
        best_result = None
        best_revenue = -np.inf
        
        for start in range(n_starts):
            # Random initialization
            p_init = np.random.rand(self.n) * self.p_max
            p_init = p_init * (self.budget / np.sum(p_init) * 0.8)
            
            # Solve from this start
            if method == 'sqp':
                result = self.solve_sequential_quadratic(p_init, max_iter=50)
            else:
                result = self.solve_projected_gradient(p_init, max_iter=100)
            
            if result['success'] and result['revenue'] > best_revenue:
                best_revenue = result['revenue']
                best_result = result
            
            print(f"  Start {start + 1}/{n_starts}: revenue = {result.get('revenue', 0):.4f}")
        
        if best_result:
            best_result['method'] = f'Multi-Start {method.upper()}'
            best_result['n_starts'] = n_starts
        
        return best_result
    
    def solve_trust_region(self, p_init: Optional[np.ndarray] = None,
                          max_iter: int = 30) -> Dict:
        """
        Method 5: Trust Region Method
        
        Uses scipy's trust-constr optimizer.
        Good balance between speed and quality.
        
        SPEED: ⭐⭐⭐⭐ (Fast)
        QUALITY: ⭐⭐⭐⭐ (Good)
        
        Args:
            p_init: Initial prices
            max_iter: Maximum iterations
            
        Returns:
            Dict with solution
        """
        print("Solving with Trust Region Method...")
        
        if p_init is None:
            p_init = np.ones(self.n) * (self.budget / self.n * 0.8)
            p_init = np.minimum(p_init, self.p_max)
        
        def objective(p):
            try:
                return -self.evaluate_revenue(p)
            except:
                return 1e10
        
        # Budget constraint: 1^T p ≤ budget
        constraints = LinearConstraint(
            np.ones((1, self.n)),
            lb=0,
            ub=self.budget
        )
        
        bounds = [(0, self.p_max[i]) for i in range(self.n)]
        
        result = minimize(
            objective,
            p_init,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'verbose': 0}
        )
        
        if result.success or result.status == 2:  # Status 2 = max iter reached
            p_opt = result.x
            y_opt, _ = self.solve_inner_problem(p_opt)
            revenue = -result.fun
            
            return {
                'p_opt': p_opt,
                'y_opt': y_opt,
                'revenue': revenue,
                'success': True,
                'method': 'Trust Region',
                'iterations': result.nit
            }
        else:
            return {'success': False, 'message': result.message}

