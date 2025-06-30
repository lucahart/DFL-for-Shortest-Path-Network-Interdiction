from typing import Tuple, Optional
from pyepo.model.opt import optModel
import torch
import numpy as np
import networkx as nx

class ShortestPath(optModel):
    """
    This class can solve shortest path problems for generic graphs.
    """

    arcs: list[tuple[int, int]]
    vertices: np.ndarray[int]
    cost: np.ndarray[float]
    graph: nx.Graph

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
            List of costs associated with each arc. If not provided, it defaults to 0.
            The length of this list should match the number of arcs.
        ------------
        Raises
        ------------
        ValueError : If the length of cost does not match the number of arcs.
        ------------
        """

        # Store arcs, vertices, and costs
        self.arcs = arcs
        self.vertices = vertices if vertices is not None else np.arange(max(max(a) for a in arcs) + 1)
        self.cost = cost if isinstance(cost, (list, np.ndarray)) else [cost]

        # Create a graph from the vertices and arcs
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertices)

        # Set cost
        self.setObj(self.cost)

        # Call the parent constructor
        super().__init__()
        pass

    def solve(self,
              source: int = 0,
              target: Optional[int] = None,
              costs: torch.Tensor | np.ndarray[float] | None = None
              ) -> Tuple[list[tuple[int, int]], float]:
        """
        Solves the shortest path problem using Dijkstra's algorithm.

        ------------
        Returns
        ------------
        path : list of tuples (int, int)
            List of arcs (edges) in the shortest path, where each arc is represented as a tuple
            of two integers (source, target).
        total_cost : float
            Total cost of the shortest path.
        ------------
        """

        # Set target to the last vertex if not provided
        if target is None:
            target = max(self.vertices)
        
        # If costs are provided, update the graph's weights
        if isinstance(costs, torch.Tensor):
            # If a tensor is provided, return a batch of solutions
            return self._solve_tensor(costs, source, target)
        elif isinstance(costs, np.ndarray):
            if costs.ndim != 1:
                raise ValueError(f"Expected costs to be a 1D array, got {costs.ndim}D array instead.")
            self.setObj(costs)
        elif costs is not None:
            raise ValueError(f"Expected costs to be a 1D array or tensor, got {type(costs)} instead.")

        # Compute the shortest path and its total cost with Dijkstra's algorithm
        solution_nodes = nx.shortest_path(self.graph, source=source, target=target, weight='weight', method='dijkstra')

        # Convert the path to a list of arcs and compute total cost
        solution_edges = [ShortestPath.sort(solution_nodes[i], solution_nodes[i+1]) for i in range(len(solution_nodes)-1)]
        objective = sum(self.graph.edges[edge]['weight'] for edge in solution_edges)

        return solution_edges, objective
    
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
            raise ValueError(f"Expected costs to be a 2D tensor, got {costs_arr.ndim}D tensor instead.")
        if costs_arr.shape[1] != len(self.arcs):
            raise ValueError(f"Expected costs to have {len(self.arcs)} columns, got {costs_arr.shape[1]} columns instead.")

        # Initialize lists to store solutions and objectives
        solutions_list = []
        objectives_list = []

        # Iterate over each instance in the batch
        for cost in costs_arr:
            # Set the costs for the current instance
            self.setObj(cost)
            sol, obj = self.solve(source=source, target=target, costs=cost)
            one_hot_sol = self.arcs_one_hot(sol)
            solutions_list.append(one_hot_sol)
            objectives_list.append(obj)
        
        # Return the solutions and objectives
        solutions = torch.from_numpy(np.array(solutions_list))
        objectives = torch.from_numpy(np.array(objectives_list))
        return solutions, objectives
    
    @staticmethod
    def sort(u: int, v: int) -> tuple[int, int]:
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
    
    def arcs_one_hot(self, 
                     arcs: list[tuple[int, int]]) -> torch.Tensor:
        """
        Converts a list of arcs to a one-hot encoded tensor.

        ------------
        Parameters
        ------------
        arcs : list of tuples (int, int)
            List of arcs (edges) in the graph, where each arc is represented as a tuple
            of two integers (source, target).
        ------------
        Returns
        ------------
        torch.Tensor
            A one-hot encoded tensor representing the arcs.
        ------------
        """
        
        # Create a one-hot encoded tensor for the arcs
        num_arcs = len(self.arcs)
        arc_indices = [self.arcs.index(arc) for arc in arcs]
        one_hot_tensor = torch.zeros(num_arcs, dtype=torch.float32)
        one_hot_tensor[arc_indices] = 1.0

        return one_hot_tensor

    def visualize(self,
                  source: int = None,
                  target: int = None,
                  color_edges: list[tuple[int, int]] | None = None,
                  dashed_edges: list[tuple[int, int]] | None = None
                  ) -> None:
        """
        TODO: Implement visualization method.
        """

        raise NotImplementedError("Visualization method is not implemented yet.")

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
        return self.graph.number_of_edges()
    
    def _getModel(self) -> nx.Graph:
        """
        Returns the underlying graph model.
        
        ------------
        Returns
        ------------
        nx.Graph
            The graph representing the shortest path model.
        ------------
        """
        return self.graph, self.cost

    def setObj(self, c: np.ndarray[float] | list[float] | float | None) -> None:
        """
        Sets the graph's weights.

        ------------
        Parameters
        ------------
        c : np.ndarray[float] | list[float] | None
            1D array or list of coefficients for the objective function. If a list/ndarray
            with a single value is provided, it is applied uniformly to all arcs.
        ------------
        """

        # Store costs
        self.costs = c if isinstance(c, (list, np.ndarray)) else [c]

        if len(c) != len(self.arcs) and len(c) != 1:
            # Check if cost length matches the number of arcs
            raise ValueError(f"cost has length {len(c)}, expected {len(self.arcs)}")
        for i, arc in enumerate(self.arcs):
            # Add edges to the graph with the specified weights
            u, v = arc
            w = c[i] if len(c) > 1 else c[0]
            self.graph.add_edge(u, v, weight=w)
        pass
