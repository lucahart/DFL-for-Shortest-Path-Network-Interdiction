from typing import Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.models.ShortestPath import ShortestPath

class ShortestPathGrid(ShortestPath):
    """
    Generic shortest path for grids.
    """

    def __init__(self, 
                m: int,
                n: int,
                cost: np.ndarray[float] | None = None
                ):
        """
        Constructs shortest path model for a grid of the size m x n.
        A cost can optionally be specified.

        ------------
        Parameters
        ------------
        m : int
            Number of rows.
        n : int
            Number of columns.
        cost : ndarray, length m*(n-1) + (m-1)*n
            Edge‐weights, first all horizontal edges (left -> right, row by row),
            then all vertical edges (top -> bottom, column by column).
        ------------
        """

        # Store grid dimensions
        self.m = m
        self.n = n

        # Create a list of nodes and edges for grid
        arcs = []
        for i in range(m):
            # edges on rows
            for j in range(n - 1):
                v = i * n + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == m - 1:
                continue
            for j in range(n):
                v = i * n + j
                arcs.append((v, v + n))
        
        # Run parent class constructor
        super().__init__(arcs, vertices=np.arange(m*n), cost=cost)
        pass

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

            # This is an implementation of the arcs_one_hot function exploiting the
            # grid structure of the graph to reduce the number of loops.

            # Find the arc indices using the grid structure
            arc_indices = []
            objective = 0.0
            for i, u in enumerate(shortest_path_nodes[:-1]):
                v = shortest_path_nodes[i + 1]
                if v == u + self.m:
                    arc_indices.append((self.n-1)*self.m + u - 1)
                else:
                    arc_indices.append((u//self.n)*(self.n-1) + u % self.n - 1)
                objective += self.cost[arc_indices[-1]]

            # Create a one-hot encoded tensor for the arcs
            one_hot_vector = np.zeros(len(self.arcs), dtype=np.float32)
            one_hot_vector[arc_indices] = 1.0

            return one_hot_vector, objective

    def visualize(self,
                color_edges: list[int, int] | None = None,
                dashed_edges: list[tuple[int, int]] | None = None
                ) -> None:
        """
        Visualize an m×n grid with edge‐weights. Optionally add
        start/end labels, highlight color edges, or highlight dashed edges.

        ------------ 
        Parameters
        ------------
        color_edges : list of tuples (int, int), optional
            Edges to highlight in color.
        dashed_edges : list of tuples (int, int), optional
            Edges to draw with dashed style.
        ------------
        """

        # Specify positions for nodes in the grid
        pos = {i: (i % self.n, 1-i // self.n) for i in range(self.m * self.n)}

        # Draw base grid
        plt.figure(figsize=(self.n, self.m))
        nx.draw(
            self.graph, pos,
            node_size=350,
            node_color='lightgray',
            edge_color='black',
            with_labels=True
        )

        # Draw weight labels if exists
        if len(self.cost) > 1:
            edge_labels = {
                edge: f"{self.cost[idx]:.2f}"
                for idx, edge in enumerate(self.arcs)
            }
            nx.draw_networkx_edge_labels(
                self.graph, pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color='black',
                label_pos=0.5
            )

        # Highlight start and end nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=[self.source],
            node_color='green'
        )
        nx.draw_networkx_nodes(
            self.graph, pos,
            nodelist=[self.target],
            node_color='red'
        )
        
        # Highlight color edges
        if color_edges is not None:
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=ShortestPath.one_hot_to_arcs(self, color_edges),
                edge_color='red',
                width=2.5
            )

        # Highlight dashed edges
        if dashed_edges is not None:
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=dashed_edges,
                style='dashed',
                width=1.5
            )
        pass
    

