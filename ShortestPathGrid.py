import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ShortestPath import ShortestPath

class ShortestPathGrid(ShortestPath):
    """
    Generic shortest path for grids.
    """

    def __init__(self, 
                m: int,
                n: int,
                cost: list[float] | np.ndarray[float] | None = None
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
        cost : sequence of float, length m*(n-1) + (m-1)*n
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

    def visualize(self,
                s: int = None,
                t: int = None,
                color_edges: list[int, int] | None = None,
                dashed_edges: list[tuple[int, int]] | None = None
                ) -> None:
        """
        Visualize an m×n grid with edge‐weights. Optionally add
        start/end labels, highlight color edges, or highlight dashed edges.

        ------------ 
        Parameters
        ------------
        s, t : int
            Start and end node indices: 0 ≤ s, t < m*n.
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
                edge: f"{self.cost[idx]}"
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
        if s is not None:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=[s],
                node_color='green'
            )
        if t is not None:
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=[t],
                node_color='red'
            )
        
        # Highlight color edges
        if color_edges is not None:
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=color_edges,
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

        # TODO: Colored edges, Dashed edges
        pass

    def get_num_edges(self) -> int:
        """
        Returns the number of edges in the graph.

        ------------
        Returns
        ------------
        int
            Number of edges in the graph.
        ------------
        """
        
        return (self.m * (self.n - 1) + (self.m - 1) * self.n)
    

