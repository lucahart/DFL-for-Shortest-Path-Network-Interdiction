from operator import __call__
import gurobipy as gp
import torch
from gurobipy import GRB
from pyepo.model.grb import optGrbModel
from typing import Tuple
from numpy import ndarray
from torch import Tensor
from copy import deepcopy
from tabulate import tabulate

from dflintdpy.models.graph import Graph
from dflintdpy.models.grid import Grid

class ShortestPathGrb(optGrbModel):

    _graph: Graph

    def __init__(self,
                 graph: Graph = None):
        
        # Store graph instance
        self._graph = deepcopy(graph)
        # Run parent class constructors
        super().__init__()
        # Update the gurobi model with the edge weights of the graph
        super().setObj(self._graph.cost)

    @classmethod
    def empty_grid(cls,
                 m: int,
                 n: int) -> 'ShortestPathGrb': 
        
        # Create an instance of Grid
        graph = Grid(m, n)
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

        return self.solve(c=cost, versatile=versatile)
    
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
        new_instance = ShortestPathGrb(deepcopy(self._graph, memo))
        return new_instance

    @property
    def cost(self):
        """
        Linear cost vector representing edge weights.
        """

        return self._graph.cost


    # TODO: Break tie if there are multiple optimal paths. Add test that we always get a single path as solution.
    def solve(self,
              c: ndarray | Tensor | None = None,
              visualize: bool = False,
              **kwargs
              ) -> Tuple[ndarray, float]:
        """
        Solve the shortest path problem.
        
        Parameters:
        -----------
        c | cost : ndarray | Tensor | None, Optional
            Cost vector for the edges. If provided, it updates the model's objective
            during the solve process. The original cost is restored after solving.
        visualize : bool, Optional
            If True, the solution is visualized in a graph. Default is False.

        Returns:
        --------
        Tuple[ndarray, float]
            A tuple containing the solution vector and the objective value.
        """
        if c is None and "cost" in kwargs:
            c = kwargs.pop("cost")

        if isinstance(c, Tensor) and c.ndim == 2:
            sols = []
            objs = []

            for row in c:
                sol, obj = self.solve(c=row, visualize=visualize, **kwargs)
                sols.append(torch.as_tensor(sol, dtype=c.dtype, device=c.device))
                objs.append(obj)

            return torch.stack(sols), torch.tensor(objs, dtype=c.dtype, device=c.device)

        # Temporarily update the objective if a new cost is provided.
        original_cost = self.cost.copy() if c is not None else None
        if c is not None:
            self.setObj(c)

        try:
            # Run solver to find solution.
            self._model.update()
            self._model.optimize()

            # Surface model failures explicitly for callers and tests.
            status = self._model.Status
            if status == GRB.INFEASIBLE:
                raise RuntimeError("Shortest path model is infeasible.")
            if status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
                raise RuntimeError("Shortest path model is unbounded.")
            if status != GRB.OPTIMAL:
                raise RuntimeError(f"Shortest path optimization failed with status {status}.")

            if isinstance(self.x, gp.MVar):
                sol = self.x.x
            else:
                sol = [self.x[k].x for k in self.x]
            obj = self._model.objVal
        finally:
            if original_cost is not None:
                self.setObj(original_cost)

        # Show solution in graph if visualize is True
        if visualize:
            self._graph.visualize(colored_edges=sol, **kwargs)

        # Return solution
        return sol, obj
    
    def evaluate(self,
                 y: ndarray | Tensor,
                 x: ndarray | Tensor | None = None
                 ) -> float | Tensor:
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
        if isinstance(y, Tensor) and y.ndim == 2:
            if y.shape[1] != self._graph.num_cost:
                raise ValueError(
                    f"Expected batched paths to have {self._graph.num_cost} columns, "
                    f"got {y.shape[1]} instead."
                )

            cost = torch.as_tensor(self.cost, dtype=y.dtype, device=y.device)

            if x is None:
                return torch.sum(y * cost, dim=1)

            intd = torch.as_tensor(x, dtype=y.dtype, device=y.device)
            if intd.ndim == 1:
                if intd.shape[0] != self._graph.num_cost:
                    raise ValueError(
                        f"Expected interdictions to have length {self._graph.num_cost}, "
                        f"got {intd.shape[0]} instead."
                    )
            elif intd.ndim == 2:
                if intd.shape != y.shape:
                    raise ValueError(
                        f"Expected batched interdictions to match path shape {tuple(y.shape)}, "
                        f"got {tuple(intd.shape)} instead."
                    )
            else:
                raise ValueError(
                    f"Expected interdictions to be 1D or 2D for batched paths, got {intd.ndim}D instead."
                )

            return torch.sum(y * (cost + intd), dim=1)

        return self._graph(y, interdictions=x)
    
    def visualize(self,
                  **kwargs):
        
        # Run visualize method of graph instance
        self._graph.visualize(**kwargs)


    # TODO: Make sure that _model is always updated when setObj or _getModel are run.
    def setObj(self,
               c: ndarray
               ) -> None:

        if isinstance(c, Tensor):
            c = c.detach().cpu().numpy()
        
        # Update local graph model objective
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

        
