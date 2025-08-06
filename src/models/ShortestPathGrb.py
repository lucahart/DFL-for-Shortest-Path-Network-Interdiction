from operator import __call__
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel
from typing import Tuple
from numpy import ndarray
from torch import Tensor
from copy import deepcopy
from tabulate import tabulate
from .ShortestPath import ShortestPath
from .ShortestPathGrid import ShortestPathGrid

class shortestPathGrb(optGrbModel):

    _graph: 'ShortestPath'

    def __init__(self,
                 graph: 'ShortestPath' = None):
        
        # Store graph instance
        self._graph = deepcopy(graph)
        # Run parent class constructors
        super().__init__()
        # Update the gurobi model with the edge weights of the graph if provided
        if graph.cost is not None:
            super().setObj(self._graph.cost)
        pass

    @classmethod
    def empty_grid(cls,
                 m: int,
                 n: int) -> 'shortestPathGrb': 
        
        # Create an instance of ShortestPathGrid
        graph = ShortestPathGrid(m, n)
        # Run parent class constructors
        return cls(graph)
    
    def __call__(self,
                 cost: ndarray | Tensor | None = None,
                 versatile: bool = False
                 ) -> Tuple[ndarray, float]:
        """
        Call method to solve the shortest path problem.
        See `solve` method for details.
        """

        return self.solve(cost = cost, versatile=versatile)
    
    def __deepcopy__(self, memo):
        """
        Create a deep copy of the shortestPathGrb instance.
        
        Parameters:
        -----------
        memo : dict
            A dictionary to keep track of already copied objects.

        Returns:
        --------
        shortestPathGrb
            A new instance of shortestPathGrb with the same attributes.
        """
        
        # Create a new instance and copy the graph
        new_instance = shortestPathGrb(deepcopy(self._graph, memo))
        return new_instance

    @property
    def cost(self):
        """
        Linear cost vector representing edge weights.
        """

        return self._graph.cost

    def solve(self,
              c: ndarray | Tensor | None = None,
              versatile: bool = False,
              **kwargs
              ) -> Tuple[ndarray, float]:
        """
        Solve the shortest path problem.
        
        Parameters:
        -----------
        cost : ndarray | Tensor | None, Optional
            Cost vector for the edges. If provided, it updates the model's objective
            during the solve process. The original cost is restored after solving.
        versatile : bool, Optional
            If True, the solution is visualized in a graph. Default is False.

        Returns:
        --------
        Tuple[ndarray, float]
            A tuple containing the solution vector and the objective value.
        """
        
        # Run solver to find solution
        sol, obj = super().solve()

        # Show solution in graph if versatile is True
        if versatile:
            self._graph.visualize(colored_edges=sol, **kwargs)

        # Return solution
        return sol, obj
    
    def evaluate(self,
                 y: ndarray | Tensor,
                 x: ndarray | Tensor | None = None
                 ) -> float:
        """
        Evaluate the objective function value for a given solution vector.

        Parameters:
        -----------
        y : ndarray | Tensor
            Solution vector representing the flow on each edge.
        x : ndarray | Tensor, Optional
            Interdiction vector representing the interdicted edges. If provided,
            it modifies the cost of the edges in the evaluation. The optimization
            model's objective is NOT updated.

        Returns:
        --------
        float
            The objective function value for the provided solution vector.
        """
        
        # Convert x to numpy array if it's a tensor
        if isinstance(y, Tensor):
            y = y.numpy()
        if isinstance(x, Tensor):
            x = x.numpy()

        return self._graph(y, interdictions=x)
    
    def visualize(self,
                  **kwargs):
        
        # Run visualize method of graph instance
        self._graph.visualize(**kwargs)

    def setObj(self,
               c: ndarray
               ) -> None:
        
        # Update local graph model objective
        if isinstance(c, Tensor):
            self._graph.setObj(c.numpy())
        else:
            self._graph.setObj(c)
        
        # Update gurobi model's objective
        super().setObj(c)
        pass

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("shortest path")
        # varibles
        x = m.addVars(self._graph.arcs, vtype=GRB.CONTINUOUS, name="x")
        # sense
        m.modelSense = GRB.MINIMIZE
        # flow conservation constraints
        for v in self._graph.vertices:
            expr = 0
            for e in self._graph.arcs:
                # flow in
                if v == e[1]:
                    expr += x[e]
                # flow out
                elif v == e[0]:
                    expr -= x[e]
            # source
            if v == self._graph.source:
                m.addConstr(expr == -1)
            # sink
            elif v == self._graph.target:
                m.addConstr(expr == 1)
            # transition
            else:
                m.addConstr(expr == 0)
        return m, x

    def pretty_print(self):
        """
        Print the Gurobi model in a human-readable format.
        This includes variables, objective function, and constraints.
        """
        
        model = self._model
        # ---------- variables ----------
        rows = [(v.VarName, v.VType, v.LB, v.UB, v.Obj)
                for v in model.getVars()]
        print("\nVariables")
        print(tabulate(rows, headers=["name", "type", "LB", "UB", "obj"]))

        # ---------- objective (assumes linear) ----------
        obj = " + ".join(f"{v.Obj:g}·{v.VarName}"
                        for v in model.getVars() if abs(v.Obj) > 1e-9)
        print("\nObjective")
        print(f" maximize {obj}\n")

        # ---------- constraints ----------
        con_rows = []
        for c in model.getConstrs():
            row  = model.getRow(c)
            expr = " + ".join(f"{row.getCoeff(i):g}·{row.getVar(i).VarName}"
                            for i in range(row.size()))
            sense = {'<': '<=', '>': '>=', '=': '='}[c.Sense]
            con_rows.append((c.ConstrName, f"{expr} {sense} {c.RHS:g}"))
        print("Constraints")
        print(tabulate(con_rows, headers=["name", "expression"]))

        
