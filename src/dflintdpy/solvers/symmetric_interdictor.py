import numpy as np
import gurobipy as gp
from gurobipy import GRB
from copy import deepcopy

from dflintdpy.models.graph import Graph
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb


class SymmetricInterdictor:
    """
    Symmetric shortest-path interdiction solver with a Gurobi leader and a
    shortest-path follower.

    The class models a two-level problem. The leader chooses up to `k` edges to
    interdict, where each chosen edge adds the corresponding entry of
    `interdiction_cost` to the follower's edge costs. The follower then solves a
    shortest-path problem on the interdicted graph. The implementation handles this
    interaction with a Benders-style loop that alternates between:

    1. Solving the follower shortest-path problem for the current edge costs.
    2. Converting the returned path into one leader scenario.
    3. Solving a max-min binary knapsack problem over all accumulated scenarios.
    4. Re-applying the chosen interdictions to the follower objective.

    Core state
    ----------
    `opt_model`
        Follower model stored as a `ShortestPathGrb` instance. It is created from a
        copied `Graph`, so the solver owns an independent follower graph state.
    `_model`
        Gurobi model used for the leader's max-min knapsack subproblem.
    `k`
        Cardinality budget limiting how many edges may be interdicted at once.
    `max_cnt`
        Maximum number of Benders iterations.
    `eps`
        Convergence tolerance for the Benders gap `z_max - z_min`.
    `interdiction_cost`
        Arc-aligned additive interdiction-cost vector used by `solve`.

    Main functionality
    ------------------
    `solve`
        Public entry point. Runs Benders decomposition with the stored
        `interdiction_cost` vector and optionally visualizes the returned follower
        path and leader interdictions.
    `benders_decomposition`
        Coordinates the alternating leader-follower loop, accumulates follower paths
        as leader scenarios, stops on epsilon convergence or iteration limit, and
        restores the original follower objective before returning.
    `solve_maxmin_knapsack`
        Solves the leader subproblem `max_x min_i (A[i] @ x + b[i])` under the
        budget `sum(x) <= k`, with `x` binary.
    `__call__`
        Convenience alias for `solve`.
    `__deepcopy__`
        Builds an independent copy of the solver, including copied follower and
        interdiction-cost state.

    Important implementation detail
    -------------------------------
    All vectors are interpreted in the follower graph's edge order. The same index
    layout is shared by the base edge costs, interdiction costs, follower path
    vectors, and leader interdiction decisions.
    """

    _LEADER_PARAM_NAMES = ("OutputFlag", "TimeLimit", "MIPGap", "Threads", "Seed")

    # Attributes
    opt_model: 'ShortestPathGrb'  # Reference to the ShortestPath object
    _model: gp.Model  # Gurobi model
    k: int  # Budget for the max-min knapsack problem
    max_cnt: int  # Maximum number of iterations for Bender's algorithm
    eps: float  # Epsilon for convergence criterion
    interdiction_cost: np.ndarray  # Interdiction cost of each edge in the graph

    def __init__(self,
                 graph: 'Graph',
                 k: int = 5,
                 interdiction_cost: np.ndarray | None = None,
                 *,
                 max_cnt: int = 10,
                 eps: float = 1,
                 output_flag: bool = False,
                 **kwargs
                 ):
        """
        Build a symmetric shortest-path interdictor with an independent follower.

        Parameters
        ----------
        graph : Graph
            Base graph for the follower shortest-path problem. The constructor wraps
            it in a new `ShortestPathGrb` instance, which deep-copies the graph so
            later mutations to the caller's `Graph` do not leak into this solver.
        k : int, optional
            Cardinality budget for the leader problem. At most `k` edges can be
            interdicted in each max-min knapsack solve.
        interdiction_cost : np.ndarray | None, optional
            Additive arc-cost vector aligned with the follower graph edge order. If
            omitted, a zero vector is created, so interdictions initially add no
            extra cost.
        max_cnt : int, optional
            Maximum number of Benders iterations.
        eps : float, optional
            Termination tolerance for the Benders gap `z_max - z_min`.
        output_flag : bool, optional
            Controls Gurobi logging on the leader model. By default logs are
            suppressed.

        Raises
        ------
        ValueError
            If `interdiction_cost` does not have one entry per graph edge.
        """

        # Copy the provided instance of a graph
        self.opt_model = ShortestPathGrb(graph)

        # Initialize the Gurobi model
        # If output_flag is False, suppress Gurobi log output
        self._model = gp.Model("maxmin_knapsack")
        if not output_flag:
            self._model.Params.OutputFlag = 0
        
        # Store available budget
        self.k = k
        # Store Bender's algorithm hyperparameters
        self.max_cnt = max_cnt
        self.eps = eps

        # Set interdiction costs if provided
        if interdiction_cost is not None:
            if len(interdiction_cost) != self.opt_model.num_cost:
                raise ValueError("Interdiction cost must match the number of edges in the graph.")
            self.interdiction_cost = deepcopy(interdiction_cost)
        else:
            # If no interdiction cost is provided, initialize with zeros
            self.interdiction_cost = np.zeros(self.opt_model.num_cost)
        pass

    def __deepcopy__(self, memo):
        """
        Create an independent copy of the interdictor and its mutable state.

        Parameters
        ----------
        memo : dict
            A dictionary to keep track of already copied objects.

        Returns
        -------
        SymmetricInterdictor
            New solver with the same budget, Benders hyperparameters, logging
            setting, follower cost vector, and interdiction-cost values, but backed
            by distinct arrays, a distinct follower solver, and a distinct copied
            graph.
        """
        
        # Create a new instance and copy the graph and other attributes
        output_flag = bool(self._model.Params.OutputFlag)
        new_instance = SymmetricInterdictor(
            deepcopy(self.opt_model._graph, memo),
            k=self.k,
            interdiction_cost=self.interdiction_cost.copy() if self.interdiction_cost is not None else None,
            max_cnt=self.max_cnt,
            eps=self.eps,
            output_flag=output_flag,
        )
        return new_instance
    
    def __call__(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Alias for :meth:`solve`.

        Returns
        -------
        interdictions_x : ndarray
            Final leader interdiction vector.
        shortest_path_y : ndarray
            Final follower shortest-path vector.
        z_min : float
            Final follower objective value.
        """

        return self.solve()

    def _reset_leader_model(self) -> None:
        """
        Rebuild the leader Gurobi model while preserving selected parameters.

        Notes
        -----
        Repeated calls to :meth:`solve_maxmin_knapsack` must start from a fresh
        model so variables and constraints from earlier scenario sets do not
        accumulate. This helper preserves only the leader settings listed in
        `_LEADER_PARAM_NAMES`.
        """

        saved_params = {
            name: getattr(self._model.Params, name)
            for name in self._LEADER_PARAM_NAMES
        }

        self._model = gp.Model("maxmin_knapsack")
        for name, value in saved_params.items():
            setattr(self._model.Params, name, value)

    def solve_maxmin_knapsack(self,
                              A: np.ndarray, 
                              b: np.ndarray, 
                              ) -> tuple[np.ndarray, float]:
        """
        Solve the leader's robust cardinality-constrained max-min problem.

        The model chooses a binary interdiction vector `x` and maximizes a scalar
        `z` subject to `sum(x) <= self.k` and `z <= A[i] @ x + b[i]` for every
        scenario row `i`. Equivalently, it maximizes the worst-case scenario value
        induced by `x`.

        Parameters
        ----------
        A : (m, n) ndarray
            Scenario matrix. Row `i` contains the coefficient vector for scenario
            `i`, and column `j` corresponds to edge `j`.
        b : (m,) ndarray
            Scenario constant terms. The method reshapes this input to 1D and
            requires one constant per scenario row in `A`.

        Returns
        -------
        x_opt : ndarray, shape (n,)
            Optimal binary interdiction vector. If multiple symmetric optima achieve
            the same worst-case value, any one of them may be returned.
        z_opt : float
            Optimal worst-case scenario value.

        Raises
        ------
        ValueError
            If `b` does not contain exactly one entry per scenario row in `A`.
        RuntimeError
            If Gurobi terminates with a non-optimal status instead of returning an
            optimal leader solution.
        """

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)

        m, n = A.shape
        if b.shape[0] != m:
            raise ValueError("b must have one entry per scenario.")

        # Rebuild the leader model on each solve so repeated calls do not
        # accumulate stale variables or constraints from earlier scenarios.
        self._reset_leader_model()

        # Decision variables
        x = self._model.addVars(n, vtype=GRB.BINARY, name="x")
        z = self._model.addVar(lb=-GRB.INFINITY, name="z")

        # Cardinality / budget constraint
        self._model.addConstr(gp.quicksum(x[j] for j in range(n)) <= self.k, name="budget")

        # Worst-case (max-min) constraints
        for i in range(m):
            expr = gp.quicksum(A[i, j] * x[j] for j in range(n)) + b[i]
            self._model.addConstr(z <= expr, name=f"scenario_{i}")

        # Objective: maximise the worst-case value z
        self._model.setObjective(z, GRB.MAXIMIZE)

        self._model.optimize()

        if self._model.Status == GRB.OPTIMAL:
            x_opt = np.array([x[j].X for j in range(n)], dtype=int)
            return x_opt, z.X
        else:
            raise RuntimeError(f"Gurobi ended with status {self._model.Status}")
        

    def benders_decomposition(self, 
                              interdiction_cost: np.ndarray,
                              versatile: bool = True
                            ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Run the alternating leader-follower Benders loop for interdiction.

        The routine first solves the follower shortest-path problem under the
        original objective. Each iteration then:

        1. Appends the current follower path as a new leader scenario with
           coefficients `interdiction_cost * shortest_path_y` and constant term
           `org_cost @ shortest_path_y`.
        2. Solves the leader max-min knapsack over all accumulated scenarios.
        3. Updates the follower objective to
           `org_cost + interdiction_cost * interdictions_x`.
        4. Resolves the follower and computes the Benders gap `z_max - z_min`.

        The loop stops once the gap is within `self.eps` or once `self.max_cnt`
        leader solves have been executed. Before returning, the follower objective is
        always restored to the original cost vector.

        Parameters
        ----------
        interdiction_cost : ndarray
            Additive arc-cost vector aligned with the follower edge order.
        versatile : bool
            If `True`, print start-up, per-iteration, and completion progress
            messages. If `False`, run silently.

        Returns
        -------
        interdictions_x : ndarray
            Last leader interdiction vector produced by the loop.
        shortest_path_y : ndarray
            Last follower shortest-path vector found before termination.
        z_min : float
            Last follower objective value found before termination.
        """

        # Print that Bender's decomposition algorithm started
        if versatile:
            print("Bender's decomposition running:\n"
                  "-------------------------------")

        # Initialization
        diff = np.inf
        cnt = 0
        org_cost = self.opt_model.cost.copy()
        interdictions_x = np.zeros_like(interdiction_cost, dtype=int)

        # Solve the shortest path problem of the follower for the first time
        shortest_path_y, z_min = self.opt_model.solve()
 
        while (diff > self.eps and cnt < self.max_cnt):
            cnt += 1
            # 
            if cnt == 1:
                A = np.reshape(interdiction_cost * shortest_path_y, (1, -1))
                b = np.array([org_cost @ shortest_path_y], dtype=float)
            else:
                A = np.vstack((A, np.reshape(interdiction_cost * shortest_path_y, (1, -1))))
                b = np.concatenate((b, np.array([org_cost @ shortest_path_y], dtype=float)))
            # Solve the max-min knapsack problem of the leader
            interdictions_x, z_max = self.solve_maxmin_knapsack(A, b)
            # Update costs
            self.opt_model.setObj(org_cost + interdiction_cost * interdictions_x)
            # Solve the shortest path of the follower
            shortest_path_y, z_min = self.opt_model.solve()
            # Calculate the difference
            diff = z_max - z_min
            if versatile:
                print(f"Iteration {cnt}: z_max = {z_max}, z_min = {z_min}")
        
        # Restore original costs
        self.opt_model.setObj(org_cost)

        if versatile:
            print("-------------------------------\n" + 
                f"Found epsilon-optimal solution after {cnt} iterations with epsilon = {diff:.2f}")

        return interdictions_x, shortest_path_y, z_min
    

    def solve(self,
              visualize: bool = False,
              versatile: bool = True,
              **kwargs
              ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the interdiction problem using the stored interdiction-cost vector.

        This is the public orchestration entry point. It forwards
        `self.interdiction_cost` into :meth:`benders_decomposition` and returns that
        method's `(interdictions_x, shortest_path_y, z_min)` tuple unchanged.

        If `visualize=True`, the follower visualization hook is called exactly once
        after the Benders loop with `colored_edges=shortest_path_y` and
        `dashed_edges=interdictions_x`, plus any extra keyword arguments.

        Parameters
        ----------
        visualize : bool, optional
            Whether to visualize the returned follower path and leader interdictions.
        versatile : bool, optional
            Whether to print Benders progress messages.
        **kwargs
            Additional keyword arguments forwarded to `self.opt_model.visualize(...)`
            when `visualize=True`.

        Returns
        -------
        interdictions_x : ndarray
            Final leader interdiction vector.
        shortest_path_y : ndarray
            Final follower shortest-path vector.
        z_min : float
            Final follower objective value.
        """

        # Compute solution
        interdictions_x, shortest_path_y, z_min = self.benders_decomposition(self.interdiction_cost, versatile=versatile)

        # Show solution in graph if visualize is True
        if visualize:
            self.opt_model.visualize(colored_edges=shortest_path_y, dashed_edges=interdictions_x, **kwargs)
            
        # Return solution
        return interdictions_x, shortest_path_y, z_min
        
