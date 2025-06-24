from typing import Tuple, Optional
import numpy as np
import networkx as nx

class ShortestPath:
    """
    This class can solve shortest path problems for generic graphs.
    """

    def __init__(self,
                arcs: list[tuple[int, int]],
                vertices: np.ndarray[int] | list[int] | None = None,
                cost: np.ndarray[float] | list[float] | None = None
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

        # Check if cost length matches the number of arcs
        if len(self.cost) != len(self.arcs) and len(self.cost) != 1:
            raise ValueError(f"cost has length {len(self.cost)}, expected {len(self.arcs)}")
        for i, arc in enumerate(self.arcs):
            u, v = arc
            w = self.cost[i] if len(self.cost) > 1 else self.cost[0]
            self.graph.add_edge(u, v, weight=w)
        pass

    def solve(self,
              source: int = 0,
              target: Optional[int] = None) -> Tuple[list[tuple[int, int]], float]:
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

        # Compute the shortest path and its total cost with Dijkstra's algorithm
        solution_nodes = nx.shortest_path(self.graph, source=source, target=target, weight='weight', method='dijkstra')

        # Convert the path to a list of arcs and compute total cost
        solution_edges = [(solution_nodes[i], solution_nodes[i+1]) for i in range(len(solution_nodes)-1)]
        objective = sum(self.graph.edges[edge]['weight'] for edge in solution_edges)

        return solution_edges, objective
        
    def visualize(self,
                  s: int = None,
                  t: int = None,
                  color_edges: list[tuple[int, int]] | None = None,
                  dashed_edges: list[tuple[int, int]] | None = None
                  ) -> None:
        """
        TODO: Implement visualization method.
        """

        raise NotImplementedError("Visualization method is not implemented yet.")
    
        pass
