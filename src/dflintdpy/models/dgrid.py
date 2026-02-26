from typing import Tuple
import numpy as np
from copy import deepcopy

from dflintdpy.models.graph import Graph
from dflintdpy.models.grid import Grid


class DGrid(Grid):
    """
    Generic shortest path for grids with diagonal arcs.
    """

    # Attributes
    m: int  # Number of rows in the grid
    n: int  # Number of columns in the grid

    def __init__(self,
                 m: int,
                 n: int,
                 cost: np.ndarray[float] | None = None
                 ):
        """
        Constructs shortest path model for a grid of the size m x n
        with diagonal arcs. A cost can optionally be specified.

        Parameters
        ----------
        m : int
            Number of rows.
        n : int
            Number of columns.
        cost : ndarray, length m*(n-1) + (m-1)*n + (m-1)*(n-1)
            Edge‐weights, first all horizontal edges (left -> right, row by row),
            then all vertical edges (top -> bottom, row by row),
            then all diagonal edges (top-left -> bottom-right, row by row).
        """

        # Store grid dimensions
        self.m = m
        self.n = n

        # Create a list of nodes and edges for grid with diagonals
        arcs = []
        for i in range(m):
            # edges on rows
            for j in range(n - 1):
                v = i * n + j
                arcs.append((v, v + 1))
            if i == m - 1:
                continue
            # edges in columns
            for j in range(n):
                v = i * n + j
                arcs.append((v, v + n))
            # diagonal edges
            for j in range(n - 1):
                v = i * n + j
                arcs.append((v, v + n + 1))

        # Run parent class constructor
        Graph.__init__(self, arcs, vertices=np.arange(m * n), cost=cost)
        pass

    def __deepcopy__(self, memo) -> 'DGrid':
        """
        Create a deepcopy of the current diagonal grid object.

        Returns
        -------
        DGrid
            A new instance of DGrid with the same properties.
        """

        return DGrid(self.m, self.n, deepcopy(self.cost, memo) if self.cost is not None else None)
    
    def scale_diagonal_edges(self, scale: float) -> None:
        """
        Scale all diagonal edges by a provided factor.

        Parameters
        ----------
        scale : float
            Scaling factor to apply to diagonal edge costs.
        """

        if not hasattr(self, 'diagonal_scale'):
            self.diagonal_scale = 1.0  # Initialize diagonal scale factor

        if scale <= 0:
            raise ValueError("scale must be positive.")

        if self.cost is None:
            raise ValueError("Diagonal scaling requires a cost vector.")

        if self.diagonal_scale == 0:
            raise ValueError("Diagonal scale is invalid and cannot be reset.")

        # Compute diagonal edge indices if not already done
        if not hasattr(self, 'diag_edge_indices'):
            idx = 0
            self.diag_edge_indices = np.zeros(self.num_cost)
            for row in range(self.m - 1):
                idx += self.n - 1 # skip horizontal edges
                idx += self.n    # skip vertical edges
                for col in range(self.n - 1):
                    self.diag_edge_indices[idx] = 1
                    idx += 1
        
        # Scale diagonal edges
        new_cost = deepcopy(self.cost)
        new_cost[self.diag_edge_indices == 1] *= scale / self.diagonal_scale
        
        # Store new diagonal scale factor
        self.diagonal_scale = scale

        # Update objective
        self.setObj(new_cost)

        return new_cost


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
        raise NotImplementedError("This method needs to be implemented correctly for DGrid.")
        # Find the arc indices using the grid structure
        arc_indices = []
        objective = 0.0
        block_size = 3 * self.n - 2
        for i, u in enumerate(shortest_path_nodes[:-1]):
            v = shortest_path_nodes[i + 1]

            row = u // self.n
            col = u % self.n
            base_idx = row * block_size

            if v == u + self.n:
                idx = base_idx + (self.n - 1) + col
            elif v == u + self.n + 1:
                idx = base_idx + (2 * self.n - 1) + col
            else:
                idx = base_idx + col

            assert self.arcs[idx] == (u, v), "Arc index mapping error"

            arc_indices.append(idx)
            objective += self.cost[idx]

        # Create a one-hot encoded tensor for the arcs
        one_hot_vector = np.zeros(len(self.arcs), dtype=np.float32)
        one_hot_vector[arc_indices] = 1.0

        return one_hot_vector, objective
