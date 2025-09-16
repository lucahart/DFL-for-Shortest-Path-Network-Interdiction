from typing import Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

from dflintdpy.models.graph import Graph

class Grid(Graph):
    """
    Generic shortest path for grids.
    """

    # Attributes
    m: int # Number of rows in the grid
    n: int # Number of columns in the grid

    def __init__(self, 
                m: int,
                n: int,
                cost: np.ndarray[float] | None = None
                ):
        """
        Constructs shortest path model for a grid of the size m x n.
        A cost can optionally be specified.

        Parameters
        ----------
        m : int
            Number of rows.
        n : int
            Number of columns.
        cost : ndarray, length m*(n-1) + (m-1)*n
            Edge‐weights, first all horizontal edges (left -> right, row by row),
            then all vertical edges (top -> bottom, column by column).
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

    def __deepcopy__(self, memo) -> 'Grid':
        """
        Create a deepcopy of the current grid object.

        Returns
        -------
        Grid
            A new instance of Grid with the same properties.
        """

        return Grid(self.m, self.n, deepcopy(self.cost, memo) if self.cost is not None else None)

    def _arcs_one_hot(self, 
                    shortest_path_nodes: list[int]
                    ) -> Tuple[np.ndarray[float], float]:
        """
        Converts a list of arcs to a one-hot encoded tensor.

        Parameters
        ----------
        shortest_path_nodes : list of integers
            List of node indices representing the shortest path.
            
        Returns
        -------
        one_hot_vector : np.ndarray[float]
            A one-hot encoded vector representing the arcs.
        objective : float
            The total cost of the shortest path represented by the one-hot vector.
        """

        # This is an implementation of the arcs_one_hot function exploiting the
        # grid structure of the graph to reduce the number of loops.

        # Find the arc indices using the grid structure
        arc_indices = []
        objective = 0.0
        for i, u in enumerate(shortest_path_nodes[:-1]):
            v = shortest_path_nodes[i + 1]

            row = u // self.n
            col = u % self.n

            if v == u + self.n:
                idx = row * (2 * self.n - 1) + (self.n - 1) + col
            else:
                idx = row * (2 * self.n - 1) + col

            assert self.arcs[idx] == (u, v), "Arc index mapping error"

            arc_indices.append(idx)
            objective += self.cost[idx]

        # Create a one-hot encoded tensor for the arcs
        one_hot_vector = np.zeros(len(self.arcs), dtype=np.float32)
        one_hot_vector[arc_indices] = 1.0

        return one_hot_vector, objective

    def visualize(self,
                *,
                colored_edges: np.ndarray | None = None,
                dashed_edges: np.ndarray | None = None,
                ax: plt.Axes | None = None,
                title: str = None,
                scale_x: float = 1.3,
                scale_y: float = 1.3,
                width: float = 1.5,
                interdictions: np.ndarray[float] | list[float] | None = None,
                **kwargs
                ) -> None:
        """
        Visualize an m×n grid with edge‐weights. Optionally add
        start/end labels, highlight color edges, or highlight dashed edges.

        Parameters
        ----------
        colored_edges : np.ndarray | None, optional
            A one-hot encoded vector representing the edges to be colored.
            If None, no edges are colored.
        dashed_edges : np.ndarray | None, optional
            A one-hot encoded vector representing the edges to be dashed.
            If None, no edges are dashed.
        ax : plt.Axes | None, optional
            The matplotlib Axes object to draw on. If None, a new figure is created.
        title : str, optional
            Title for the plot. If None, no title is set.
        scale_x : float, optional
            Scaling factor for the x-axis. Default is 1.3.
        scale_y : float, optional
            Scaling factor for the y-axis. Default is 1.3.
        width : float, optional
            Width of the edges in the plot. Default is 1.5.
        interdictions : np.ndarray[float] | list[float] | None, optional
            A vector representing the additional interdiction costs on all edges.
            If None, no additional interdiction costs are shown.    
        """

        # Specify positions for nodes in the grid
        pos = {i: (i % self.n, 1-i // self.n) for i in range(self.m * self.n)}

        # Draw base grid
        if ax is None:
            plt.figure(figsize=(self.n*scale_x, self.m*scale_y))
            ax = plt.gca()
            ax.set_axis_off()
        else:
            plt.sca(ax)
            plt.axis('off')

        edge_artists = nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color="black", width=width, **kwargs)
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color="w", edgecolors="k")
        nx.draw_networkx_labels(self.graph, pos, ax=ax)

        # Draw weight labels if exists
        if len(self.cost) > 1:
            if interdictions is not None:
                edge_labels = {
                    edge: f"{self.cost[idx]:.2f} + {interdictions[idx]:.2f}" 
                    if interdictions[idx] > .01 
                    else f"{self.cost[idx]:.2f}"
                    for idx, edge in enumerate(self.arcs)
                }
            else:
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
        if colored_edges is not None:
            for patch, e in zip(edge_artists, self.graph.edges()):
                new_color = "red" if e in Graph.one_hot_to_arcs(self, colored_edges) else "black"
                patch.set_color(new_color)

        # Highlight dashed edges
        if dashed_edges is not None:
            for patch, e in zip(edge_artists, self.graph.edges()):
                new_line_style = "dashed" if e in Graph.one_hot_to_arcs(self, dashed_edges) else "solid"
                patch.set_linestyle(new_line_style)

        if title is not None:
            ax.set_title(title)

        pass
