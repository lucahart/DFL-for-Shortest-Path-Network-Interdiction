from typing import Tuple
from pyepo.model.opt import optModel
import torch
import numpy as np
import networkx as nx
from copy import deepcopy

class ShortestPath(optModel):
    """
    This class can solve shortest path problems for generic graphs.
    """

    # Attributes
    arcs: list[tuple[int, int]] # list of arcs (edges) in the graph
    vertices: np.ndarray[int] # list of vertices (nodes) in the graph
    cost: np.ndarray[float] # list of costs associated with each arc
    graph: nx.Graph # networkx graph representation
    source: int # source node for the shortest path
    target: int # target node for the shortest path

    def __init__(self,
                arcs: list[tuple[int, int]],
                vertices: np.ndarray[int] | None = None,
                cost: np.ndarray[float] | None = None
                ) -> None:
        """
        Constructor for shortest path class.

        ------------
        Parameters
        ------------
        arcs : list of tuples (int, int)
            List of arcs (edges) in the graph, where each arc is represented as a tuple
            of two integers (source, target).
        vertices : np.ndarray[int] | list[int], optional
            List of vertices (nodes) in the graph. If not provided, it defaults to a
            range of integers from 0 to the maximum vertex index found in arcs.
        cost : np.ndarray[float] | list[float], optional
            List of costs associated with each arc. 
            If not provided, it defaults to 1 for all arcs.
            The length of this list should match the number of arcs.
        ------------
        Raises
        ------------
        ValueError : If the length of cost does not match the number of arcs.
        ------------
        """

        # Store arcs, vertices, and costs
        self.arcs = arcs
        self.vertices = (vertices 
                         if vertices is not None 
                         else np.arange(
                             max(max(a) for a in arcs) + 1
                             )
                        )

        # Create a graph from the vertices and arcs
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.vertices)

        # Set cost
        if cost is not None:
            self.setObj(cost)
        else:
            self.setObj(np.ones(len(arcs), dtype=float))

        # Call the parent constructor
        super().__init__()
        pass

    def __deepcopy__(self, memo) -> 'ShortestPath':
        """
        Creates a deepcopy of the current ShortestPath instance.
        
        Parameters
        ----------
        memo : dict
            A dictionary to keep track of already copied objects.

        Returns
        -------
        ShortestPath
            A new instance of ShortestPath with the same attributes.
        """

        new_instance = ShortestPath(
            arcs=deepcopy(self.arcs, memo),
            vertices=deepcopy(self.vertices, memo),
            cost=deepcopy(self.cost, memo)
        )
        return new_instance
    
    def __call__(self,
                 path: np.ndarray[float]) -> float:
        """
        Call method to compute the cost of a given path.

        Parameters
        ----------
        path : np.ndarray[float]
            A one-hot encoded vector representing the arcs in the path.

        Returns
        -------
        float
            The total cost of the path represented by the one-hot vector.
        """

        return path @ self.cost

    def solve(self,
              cost: torch.Tensor | np.ndarray[float] | None = None
              ) -> Tuple[np.ndarray, float]:
        """
        Solves the shortest path problem using Dijkstra's algorithm.

        ------------
        Returns
        ------------
        shortest_path : list of tuples (int, int)
            List of arcs (edges) in the shortest path, where each arc is 
            represented as a tuple of two integers (source, target).
        objective : float
            Total cost of the shortest path.
        ------------
        """
        
        # If costs are provided, update the graph's weights
        if isinstance(cost, torch.Tensor):
            # If a tensor is provided, return a batch of solutions
            return self._solve_tensor(cost, self.source, self.target)
        elif isinstance(cost, np.ndarray):
            self.setObj(cost)
        elif cost is not None:
            raise ValueError(
                f"Expected costs to be a 1D array or tensor, got {type(cost)} instead.")

        # Compute the shortest path and its total cost with Dijkstra's algorithm
        shortest_path_nodes = nx.shortest_path(
            self.graph, 
            source=self.source, 
            target=self.target, 
            weight='weight', 
            method='dijkstra'
            )

        # Convert the path to a one-hot vector representation

        shortest_path, objective = self._arcs_one_hot(shortest_path_nodes)

        return shortest_path, objective
    
    def _solve_tensor(self, 
                      costs: torch.Tensor,
                      source: int,
                      target: int) -> None:
        """
        Solves a batch of shortest path problems using the provided costs tensor.

        ------------
        Parameters
        ------------
        costs : Tensor[float]
            A tensor containing the costs for all instances of the data batch.
        ------------
        """
        
        # Ensure costs is a 2D tensor of appropriate shape
        costs_arr = costs.detach().numpy()
        if costs_arr.ndim != 2:
            raise ValueError(
                f"Expected costs to be a 2D tensor, got {costs_arr.ndim}D tensor instead.")
        if costs_arr.shape[1] != len(self.arcs):
            raise ValueError(
                f"Expected costs to have {len(self.arcs)} columns, " + 
                f"got {costs_arr.shape[1]} columns instead.")

        # Initialize lists to store solutions and objectives
        solutions_list = []
        objectives_list = []

        # Iterate over each instance in the batch
        for cost in costs_arr:
            # Set the costs for the current instance
            self.setObj(cost)
            sol, obj = self.solve(source=source, target=target, cost=cost)
            one_hot_sol = self._arcs_one_hot(sol)
            solutions_list.append(one_hot_sol)
            objectives_list.append(obj)
        
        # Return the solutions and objectives
        solutions = torch.from_numpy(np.array(solutions_list))
        objectives = torch.from_numpy(np.array(objectives_list))
        return solutions, objectives
    
    @staticmethod
    def __sort(u: int, v: int) -> tuple[int, int]:
        """
        Sorts the arc (u, v) in ascending order.

        ------------
        Parameters
        ------------
        u : int
            Source vertex of the arc.
        v : int
            Target vertex of the arc.
        ------------
        Returns
        ------------
        tuple[int, int]
            A tuple representing the sorted arc (min(u, v), max(u, v)).
        ------------
        """
        return (min(u, v), max(u, v))
    
    def _arcs_one_hot(self, 
                     shortest_path_nodes: list[int]
                     ) -> Tuple[np.ndarray[float], float]:
        """
        Converts a list of arcs to a one-hot encoded tensor.

        ------------
        Parameters
        ------------
        shortest_path_nodes : list of integers
            List of node indices representing the shortest path.
        ------------
        Returns
        ------------
        one_hot_vector : np.ndarray[float]
            A one-hot encoded vector representing the arcs.
        objective : float
            The total cost of the shortest path represented by the one-hot vector.
        ------------
        """

        # Create list of arcs form shortest path nodes
        shortest_path = [ShortestPath.__sort(shortest_path_nodes[i],
                                              shortest_path_nodes[i + 1]
                                              )
                         for i in range(len(shortest_path_nodes) - 1)]
        # objective = sum(self.graph.edges[edge]['weight'] for edge in shortest_path)

        # Create a one-hot encoded tensor for the arcs
        num_arcs = len(self.arcs)
        arc_indices = [self.arcs.index(arc) for arc in shortest_path]
        one_hot_vector = np.zeros(num_arcs, dtype=np.float32)
        one_hot_vector[arc_indices] = 1.0
        objective = one_hot_vector @ self.cost

        return one_hot_vector, objective

    def visualize(self,
                  colored_edges: np.ndarray | None = None,
                  dashed_edges: np.ndarray | None = None
                  ) -> None:
        """
        TODO: Implement visualization method.
        """

        raise NotImplementedError("Visualization method is not implemented yet.")
    
    def _getModel(self):
        # """
        # A method to build Gurobi model (from PyEPO Documentation).

        # Returns:
        #     tuple: optimization model and variables
        # """
        # import gurobipy as gp
        # from gurobipy import GRB
        # # ceate a model
        # m = gp.Model("shortest path")
        # # varibles
        # x = m.addVars(self.arcs, name="x")
        # # sense
        # m.modelSense = GRB.MINIMIZE
        # # flow conservation constraints
        # for i in range(self.grid[0]):
        #     for j in range(self.grid[1]):
        #         v = i * self.grid[1] + j
        #         expr = 0
        #         for e in self.arcs:
        #             # flow in
        #             if v == e[1]:
        #                 expr += x[e]
        #             # flow out
        #             elif v == e[0]:
        #                 expr -= x[e]
        #         # source
        #         if i == 0 and j == 0:
        #             m.addConstr(expr == -1)
        #         # sink
        #         elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
        #             m.addConstr(expr == 1)
        #         # transition
        #         else:
        #             m.addConstr(expr == 0)
        # return m, x

        return self.graph, self.cost

    def setObj(self, 
               c: np.ndarray[float] | list[float] | float | None,
               source: int = 0,
               target: int = None
               ) -> None:
        """
        Sets the graph's weights.

        ------------
        Parameters
        ------------
        c : np.ndarray[float] | list[float] | None
            1D array or list of coefficients for the objective function. If a list/ndarray
            with a single value is provided, it is applied uniformly to all arcs.
        source : int, optional
            The source node for the shortest path. Defaults to 0.
        target : int, optional
            The target node for the shortest path. 
            If not provided, defaults to the last vertex.
        ------------
        """
        # Set source and target
        self.source = source

        # Set target to the last vertex if not provided
        if target is None or target < 0 or target > max(self.vertices):
            self.target = max(self.vertices)
        else:
            self.target = target

        # Check if cost is in the correct data format
        if isinstance(c, (list, np.ndarray)):
            cost = np.squeeze(c)
        else:
            raise ValueError(
                f"Expected cost to be ndarray or list, got {type(c)} instead."
            )
        # Check if cost is a 1D arrays
        if cost.ndim != 1:
            raise ValueError(
                f"Expected costs to be a 1D array, got {cost.ndim}D array instead."
            )
        # Check if the length of cost matches the number of arcs
        if len(c) != len(self.arcs) and len(c) != 1:
            raise ValueError(f"cost has length {len(c)}, expected {len(self.arcs)}")
        # Store cost attribute if all checks pass
        self.cost = cost

        # Add edges to the graph with the specified weights
        for i, arc in enumerate(self.arcs):
            u, v = arc
            w = c[i] if len(c) > 1 else c[0]
            self.graph.add_edge(u, v, weight=w)
        pass

    @staticmethod
    def one_hot_to_arcs(model: 'ShortestPath',
                         one_hot_vector: np.ndarray[float]
                         ) -> list[tuple[int, int]]:
        """
        Converts a one-hot encoded vector back to a list of arcs.

        ------------
        Parameters
        ------------
        model : ShortestPath
            The ShortestPath model instance.
        one_hot_vector : np.ndarray[float]
            A one-hot encoded vector representing the arcs.
        ------------
        Returns
        ------------
        arcs : list of tuples (int, int)
            List of arcs (edges) in the graph, where each arc is represented as a tuple
            of two integers.
        ------------
        """
        return [arc for arc in model.arcs if one_hot_vector[model.arcs.index(arc)] > 0]

    @property
    def num_edges(self) -> int:
        """
        Returns the number of edges in the graph.

        ------------
        Returns
        ------------
        int
            Number of edges in the graph.
        ------------
        """
        return len(self.arcs)
