import numpy as np
import gurobipy as gp
from gurobipy import GRB
from copy import deepcopy

from src.models.ShortestPath import ShortestPath
from src.models.ShortestPathGrb import shortestPathGrb


class BendersDecomposition:

    # Attributes
    opt_model: 'ShortestPath'  # Reference to the ShortestPath object
    _model: gp.Model  # Gurobi model
    k: int  # Budget for the max-min knapsack problem
    max_cnt: int  # Maximum number of iterations for Bender's algorithm
    eps: float  # Epsilon for convergence criterion
    interdiction_cost: np.ndarray  # Cost of interdicting each edge in the graph
    # n: int  # Number of arcs (edges) in the grid (also number of decision variables in the max-min knapsack problem)

    def __init__(self,
                 opt_model: 'ShortestPath',
                 k: int = 5,
                 interdiction_cost: np.ndarray | None = None,
                 max_cnt: int = 10,
                 eps: float = 1,
                 output_flag: bool = False
                 ):

        # Copy the provided instance of a graph
        self.opt_model = deepcopy(opt_model)

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
            self.interdiction_cost = interdiction_cost
        else:
            # If no interdiction cost is provided, initialize with zeros
            self.interdiction_cost = np.zeros(self.opt_model.num_cost)
        pass

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the BendersDecomposition instance.
        
        Parameters
        ----------
        memo : dict
            A dictionary to keep track of already copied objects.

        Returns
        -------
        BendersDecomposition
            A new instance of BendersDecomposition with the same attributes.
        """
        
        # Create a new instance and copy the graph and other attributes
        new_instance = BendersDecomposition(self.opt_model, 
                                            self.k, 
                                            self.interdiction_cost.copy() if self.interdiction_cost is not None else None,
                                            self.max_cnt, 
                                            self.eps)
        return new_instance
    
    def __call__(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Call the Benders decomposition method.
        
        Returns
        -------
        interdictions_x : ndarray
            Optimal decision vector from the max-min knapsack problem.
        shortest_path_y : ndarray
            Decision vector from the shortest path problem.
        z_min : float
            The minimum cost of the shortest path.
        """

        return self.solve()

    def solve_maxmin_knapsack(self,
                              A: np.ndarray, 
                              b: np.ndarray, 
                              ) -> tuple[np.ndarray, float]:
        """
        Robust (max-min) knapsack with a cardinality budget.
        
        Parameters
        ----------
        A : (m, n) ndarray
            Row i contains the coefficient vector a_i^T of scenario i.
        b : (m,) ndarray
            Constant terms b_i for each scenario.
        k : int
            Budget on the number of items that can be selected.
        output_flag : bool, optional
            If False, suppresses Gurobi log output.
        
        Returns
        -------
        x_opt : ndarray, shape (n,)
            Binary decision vector (1 if item j is chosen, else 0).
        z_opt : float
            Optimal objective value.
        """

        m, n = A.shape

        # Decision variables
        x = self._model.addVars(n, vtype=GRB.BINARY, name="x")
        z = self._model.addVar(lb=-GRB.INFINITY, name="z")

        # Cardinality / budget constraint
        self._model.addConstr(gp.quicksum(x[j] for j in range(n)) <= self.k, name="budget")

        # Worst-case (max-min) constraints
        for i in range(m):
            expr = gp.quicksum(A[i, j] * x[j] for j in range(n)) + float(b[i])
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
                              interdiction_cost
                            ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Perform Benders decomposition for the given grid and interdiction cost.
        
        Parameters
        ----------
        grid : ShortestPathGrid
            The grid object containing the graph and costs.
        interdiction_cost : ndarray
            The cost of interdicting each edge.

        Returns
        -------
        interdictions_x : ndarray
            Optimal decision vector from the max-min knapsack problem.
        shortest_path_y : ndarray
            Decision vector from the shortest path problem.
        diff : float
            The difference between the max-min and min-max objective values.
        """

        # Print that Bender's decomposition algorithm started
        print("Bender's decomposition running:\n"
              "-------------------------------")

        # Initialization
        diff = np.inf
        cnt = 0
        org_cost = self.opt_model.cost.copy()
 
        while (diff > self.eps and cnt < self.max_cnt):
            cnt += 1
            # Solve the shortest path problem of the follower
            shortest_path_y, z_min = self.opt_model.solve()
            # 
            if cnt == 1:
                A = np.reshape(interdiction_cost * shortest_path_y, (1, -1))
                b = np.reshape(org_cost @ shortest_path_y, 1)
            else:
                A = np.vstack((A, np.reshape(interdiction_cost * shortest_path_y, (1, -1))))
                b = np.vstack((b, np.reshape(org_cost @ shortest_path_y, 1)))
            # Solve the max-min knapsack problem of the leader
            interdictions_x, z_max = self.solve_maxmin_knapsack(A, b)
            # Update costs
            self.opt_model.setObj(org_cost + interdiction_cost * interdictions_x)
            # Calculate the difference
            diff = z_max - z_min
            print(f"Iteration {cnt}: z_max = {z_max}, z_min = {z_min}")
        
        # Restore original costs
        self.opt_model.setObj(org_cost)

        print("-------------------------------\n" + 
              f"Found epsilon-optimal solution after {cnt} iterations with epsilon = {diff:.2f}")

        return interdictions_x, shortest_path_y, z_min
    
    def solve(self,
              versatile: bool = False,
              **kwargs
              ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the Benders decomposition problem.
        
        Returns
        -------
        interdictions_x : ndarray
            Optimal decision vector from the max-min knapsack problem.
        shortest_path_y : ndarray
            Decision vector from the shortest path problem.
        z_min : float
            The minimum cost of the shortest path.
        """

        # Compute solution
        interdictions_x, shortest_path_y, z_min = self.benders_decomposition(self.interdiction_cost)

        # Show solution in graph if versatile is True
        if versatile:
            self.opt_model.visualize(colored_edges=shortest_path_y, dashed_edges=interdictions_x, **kwargs)
            
        # Return solution
        return interdictions_x, shortest_path_y, z_min
        

