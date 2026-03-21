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
    """
    Gurobi-backed shortest-path solver built around a deep-copied ``Graph``.

    Main state:
    - ``_graph`` stores the local graph copy, including arcs, vertices, source,
      target, and the current edge-cost vector.
    - ``_model`` and ``x`` are created by ``optGrbModel`` through ``_getModel``;
      they represent the live Gurobi model and one continuous decision variable
      per arc.
    - ``cost`` exposes the currently stored graph cost vector.

    Core functionality:
    - ``__init__(graph)`` copies the input graph, builds the source-target flow
      model, and immediately pushes the graph cost into the Gurobi objective so
      the solver state and model objective start in sync.
    - ``solve(c=None, visualize=False, **kwargs)`` optimizes the current model.
      It can temporarily replace the objective with a provided cost vector,
      supports batched 2D tensor costs by solving one row at a time, restores
      the original objective after temporary solves, raises explicit errors for
      infeasible or unbounded models, and can visualize the selected path.
    - ``evaluate(y, x=None)`` scores a candidate path/flow without re-solving.
      For 1D inputs it delegates to ``Graph.evaluate``; for batched tensor paths
      it returns one objective per row and optionally adds shared or batched
      interdiction vectors.
    - ``setObj(c)`` updates both the graph's stored costs and the live Gurobi
      objective coefficients, accepting lists, NumPy arrays, and tensors.

    Supporting methods:
    - ``empty_grid(m, n)`` builds a solver on a new ``Grid`` instance.
    - ``__call__`` forwards directly to ``solve``.
    - ``__deepcopy__`` rebuilds an equivalent solver around a copied graph.
    - ``visualize`` delegates drawing to the underlying graph.
    - ``_getModel`` creates the minimum-cost flow formulation with one flow
      balance constraint per vertex.
    - ``pretty_print`` prints the current model variables, objective, and
      constraints in a readable tabular form.
    """

    _graph: Graph

    def __init__(self,
                 graph: Graph = None):
        """
        Build a shortest-path solver from a graph-like object.

        Parameters
        ----------
        graph : Graph, optional
            Graph instance supplying arcs, vertices, source, target, and edge
            costs. The graph is deep-copied so later solver-side objective
            updates do not mutate the caller's object.

        Notes
        -----
        Construction performs three linked steps:
        1. Copy the input graph into ``self._graph``.
        2. Let ``optGrbModel`` build the Gurobi model and decision variables by
           calling ``_getModel``.
        3. Push the copied graph cost vector into the live Gurobi objective so
           the internal graph state and the optimization model start aligned.
        """
        
        # Store graph instance
        self._graph = deepcopy(graph)
        # Run parent class constructors
        super().__init__()
        # Update the gurobi model with the edge weights of the graph
        self.setObj(self._graph.cost)

    @classmethod
    def empty_grid(cls,
                 m: int,
                 n: int) -> 'ShortestPathGrb': 
        """
        Create a solver on a newly constructed ``Grid`` graph.

        Parameters
        ----------
        m, n : int
            Grid dimensions.

        Returns
        -------
        ShortestPathGrb
            Solver initialized on ``Grid(m, n)`` with the grid's default costs.
        """
        
        # Create an instance of Grid
        graph = Grid(m, n)
        # Run parent class constructors
        return cls(graph)
    
    def __call__(self,
                 cost: ndarray | Tensor | None = None,
                 versatile: bool = False,
                 **kwargs
                 ) -> Tuple[ndarray, float]:
        """
        Solve through function-call syntax.

        Parameters
        ----------
        cost : ndarray | Tensor | None, optional
            Temporary cost vector or batched tensor cost matrix forwarded as
            ``c`` to ``solve``.
        versatile : bool, optional
            Forwarded unchanged to ``solve`` for compatibility with the parent
            interface.

        Returns
        -------
        tuple
            Same return contract as ``solve``:
            - 1D or scalar-cost solve -> ``(solution, objective)`` where
              ``solution`` is array-like and ``objective`` is a float.
            - 2D tensor-cost solve -> ``(solutions, objectives)`` where both are
              tensors with one row or value per cost row.
        """

        return self.solve(c=cost, versatile=versatile, **kwargs)
    
    def __deepcopy__(self, memo):
        """
        Rebuild an equivalent solver around a copied graph.

        Parameters
        ----------
        memo : dict
            Standard ``deepcopy`` memo dictionary.

        Returns
        -------
        ShortestPathGrb
            New solver instance with an independent copied graph, Gurobi model,
            and objective coefficients.
        """
        
        # Create a new instance and copy the graph
        new_instance = ShortestPathGrb(deepcopy(self._graph, memo))
        return new_instance

    @property
    def cost(self):
        """
        Return the current edge-cost vector stored on the local graph copy.

        Returns
        -------
        ndarray
            One cost coefficient per arc in ``self._graph.arcs``.
        """

        return self._graph.cost


    # TODO: Break tie if there are multiple optimal paths. Add test that we always get a single path as solution.
    def solve(self,
              c: ndarray | Tensor | None = None,
              visualize: bool = False,
              **kwargs
              ) -> Tuple[ndarray, float]:
        """
        Optimize the current shortest-path model.

        Parameters
        ----------
        c : ndarray | Tensor | None, optional
            Temporary objective coefficients for the solve. If provided, the
            solver updates the graph and Gurobi objective before optimizing, then
            restores the original objective afterward.
        visualize : bool, optional
            If ``True``, draw the solved path on the underlying graph after a
            successful optimization.
        **kwargs
            Additional options. The method also accepts ``cost=...`` as an alias
            for ``c``. If both ``c`` and ``cost`` are supplied, ``c`` takes
            precedence and ``cost`` is ignored.
            Additional parameters for visualization can also be passed through 
            ``**kwargs`` and are forwarded to the graph's ``visualize`` method 
            if ``visualize=True``.

        Returns
        -------
        tuple[np.ndarray | list, float] | tuple[Tensor, Tensor]
            Return shape depends on the cost input:
            - No cost override, list, 1D NumPy array, or 1D tensor:
              returns ``(sol, obj)`` for one optimization, where ``sol`` is the
              one-hot solution as a numpy array and ``obj`` is a Python float.
            - 2D tensor cost matrix:
              treats each row as one temporary objective and returns
              ``(sols, objs)`` where both are tensors with one solution and one
              objective per row.

        Raises
        ------
        RuntimeError
            If the model is infeasible, unbounded, or terminates with a
            non-optimal status.

        Notes
        -----
        Batched solves are implemented by iterating over the rows of a 2D tensor
        and reusing the same model with temporary objective updates.
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
        Evaluate the objective value of a candidate path or flow.

        Parameters
        ----------
        y : ndarray | Tensor
            Candidate edge-flow vector. A 1D list/array/tensor is treated as one
            path. A 2D tensor is treated as a batch with one path per row.
        x : ndarray | Tensor | None, optional
            Optional interdiction adjustment added to the base cost during
            evaluation only. It does not modify the stored graph cost or Gurobi
            objective. For batched tensor paths, ``x`` may be one shared 1D
            interdiction vector or a 2D tensor matching ``y``.

        Returns
        -------
        float | Tensor
            - 1D input path -> scalar objective value.
            - 2D tensor path batch -> 1D tensor with one objective per row.

        Raises
        ------
        ValueError
            If batched path or interdiction shapes are incompatible.

        Notes
        -----
        For non-batched inputs, evaluation is delegated to the underlying graph
        object rather than the Gurobi model.
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

        return self._graph.evaluate(y, interdictions=x)
    
    def visualize(self,
                  **kwargs):
        """
        Forward graph-visualization requests to the underlying graph object.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed directly to ``self._graph.visualize``.
        """
        
        # Run visualize method of graph instance
        self._graph.visualize(**kwargs)


    # TODO: Make sure that _model is always updated when setObj or _getModel are run.
    def setObj(self,
               c: ndarray
               ) -> None:
        """
        Replace the stored objective coefficients on both graph and model.

        Parameters
        ----------
        c : list | ndarray | Tensor
            New edge-cost vector. Tensor inputs are detached and converted to a
            NumPy array before updating the graph and Gurobi model.

        Notes
        -----
        This method updates both layers of state:
        - ``self._graph.cost`` via ``Graph.setObj``.
        - The live Gurobi variable objective coefficients via the parent
          ``optGrbModel.setObj`` implementation.

        The model is updated immediately so the next ``solve`` uses the new
        objective without rebuilding the model.
        """

        if isinstance(c, Tensor):
            c = c.detach().cpu().numpy()
        
        # Update local graph model objective
        self._graph.setObj(c)
        
        # Update gurobi model's objective
        super().setObj(c)
        self._model.update()
        pass

    def _getModel(self):
        """
        Build the Gurobi shortest-path model.

        Returns
        -------
        tuple[gp.Model, gp.tupledict]
            Model and arc-variable mapping used by ``optGrbModel``.

        Notes
        -----
        The formulation creates one continuous flow variable per arc, sets a
        minimization objective, and adds one flow-balance equality per vertex:
        source has net outflow 1, target has net inflow 1, and intermediate
        vertices have zero net flow.
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
        Print the current Gurobi model in a readable tabular form.

        The output includes variable metadata, the linear objective, and every
        stored constraint with its assembled row expression.
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
        print(f" minimize {obj}\n")

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

        
