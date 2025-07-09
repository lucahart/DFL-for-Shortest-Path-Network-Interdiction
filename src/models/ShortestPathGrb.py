import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel
from typing import Tuple
from numpy import ndarray
from src.models.ShortestPath import ShortestPath

class shortestPathGrb(optGrbModel):

    _graph: 'ShortestPath'

    def __init__(self,
                 graph: 'ShortestPath'):
        
        # Store graph instance
        self._graph = graph
        # Run parent class constructors
        super().__init__()
        # Update the gurobi model with the edge weights of the graph
        self.setObj(self._graph.cost)
        pass

    @property
    def cost(self):
        """
        Linear cost vector representing edge weights.
        """

        return self._graph.cost
    
    def solve(self,
              versatile: bool = False
              ) -> Tuple[ndarray, float]:
        
        # Run solver to find solution
        sol, obj = super().solve()

        # Show solution in graph if versatile is True
        if versatile:
            self._graph.visualize(colored_edges=sol)

        # Return solution
        return sol, obj
    
    def visualize(self,
                  colored_edges: ndarray | None = None,
                  dashed_edges: ndarray | None = None):
        
        # Run visualize method of graph instance
        self._graph.visualize(colored_edges=colored_edges, 
                              dashed_edges=dashed_edges)

    def setObj(self,
               c: ndarray
               ) -> None:
        
        self._graph.setObj(c)
        super().setObj(c)

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("shortest path")
        # varibles
        x = m.addVars(self._graph.arcs, name="x")
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
    
    def copy(self):

        return shortestPathGrb(self._graph)