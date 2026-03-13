from importlib.resources import path
from typing import Tuple
from pyepo.model.opt import optModel
import torch
import numpy as np
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt

class Graph(optModel):
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
                cost: np.ndarray[float] | None = None,
                source: int = None,
                target: int = None
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

        # Store arcs and vertices
        self.arcs = arcs
        if vertices is None:
            self.vertices = np.arange(max(max(a) for a in arcs) + 1)
        else:
            self.vertices = vertices
        # TODO: Currently assums that vertices are labeled from 0 to n-1, 
        # need to change that if we want to allow for arbitrary vertex labels

        # Create a graph from the vertices and arcs
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.vertices)

        # Set cost
        if cost is not None:
            self.setObj(cost, source=source, target=target)
        else:
            self.setObj(np.ones(len(arcs), dtype=float))
        
        # Set source and target nodes if provided
        self._set_source_target(source, target)

        # Call the parent constructor
        super().__init__()
        pass

    def __deepcopy__(self, memo) -> 'Graph':
        """
        Creates a deepcopy of the current ShortestPath instance.
        
        Parameters
        ----------
        memo : dict
            A dictionary to keep track of already copied objects.

        Returns
        -------
        Graph
            A new instance of Graph with the same attributes.
        """

        new_instance = Graph(
            arcs=deepcopy(self.arcs, memo),
            vertices=deepcopy(self.vertices, memo),
            cost=deepcopy(self.cost, memo)
        )
        return new_instance
    
    def __call__(self,
                 path: np.ndarray[float],
                 interdictions: np.ndarray[float] | None = None
                 ) -> float:
        """
        Call method to compute the cost of a given path.

        Parameters
        ----------
        path : np.ndarray[float]
            A one-hot encoded vector representing the arcs in the path.
        interdictions : np.ndarray[float] | None, optional
            A vector representing the interdiction values on the arcs. 
            The graph model's objective is NOT updated.

        Returns
        -------
        float
            The total cost of the path represented by the one-hot vector.
        """

        return self.evaluate(path, interdictions)
    
    def evaluate(self,
                 path: np.ndarray[float],
                 interdictions: np.ndarray[float] | None = None
                 ) -> float:
        """
        Evaluation method to compute the cost of a given path.

        Parameters
        ----------
        path : np.ndarray[float]
            A one-hot encoded vector representing the arcs in the path.
        interdictions : np.ndarray[float] | None, optional
            A vector representing the interdiction values on the arcs. 
            The graph model's objective is NOT updated.

        Returns
        -------
        float
            The total cost of the path represented by the one-hot vector.
        """

        # Convert provided path to numpy array
        new_path = self._to_1d_numpy(path)

        # Return the objective value if no interdictions are provided
        if interdictions is None:
            return new_path @ self.cost
        
        # Convert provided interdictions to numpy array
        new_interdictions = self._to_1d_numpy(interdictions)

        return new_path @ (self.cost + new_interdictions)
    
    def _to_1d_numpy(
            self, 
            vector: np.ndarray[float] | torch.Tensor | list[float]
        ) -> np.ndarray[float]:
        """
        Converts the provided list, ndarray, or tensor to a numpy array.
        Checks that:
        - The input is one of the types: numpy array, torch tensor, or list.
        - The resulting array is 1D.
        - The resulting array length matches the number of arcs in the graph.

        Parameters
        ----------
        vector : np.ndarray[float] | torch.Tensor | list[float]
            The vector to be converted, which can be a numpy array, torch tensor, or list.

        Returns
        -------
        np.ndarray[float]
            The vector converted to a numpy array.

        Raises
        ------
        TypeError
            If the input vector is not a numpy array, torch tensor, or list.
        ValueError
            If the resulting array is not 1D.
            If the resulting array length does not match the number of arcs in the graph.
        """
        # Convert vector to numpy array if it's a torch tensor or list.
        # Raise error if it's not one of the expected types.
        if isinstance(vector, torch.Tensor):
            new_vector = vector.detach().numpy().squeeze()
        elif isinstance(vector, np.ndarray):
            new_vector = vector.squeeze()
        elif isinstance(vector, list):
            new_vector = np.array(vector).squeeze()
        else:
            raise TypeError(f"Expected vector to be a numpy array, " + 
                            f"torch tensor, or list, got {type(vector)} instead.")

        # Check that the vector is 1D and has the correct length
        if new_vector.ndim != 1:
            raise ValueError(f"Expected vector to be a 1D array, " + 
                             f"got {new_vector.ndim}D array instead.")
        if len(new_vector) != len(self.arcs):
            raise ValueError(f"Expected vector to have length {len(self.arcs)}," +
                             f" got {len(new_vector)} instead.")
        
        # Return the converted vector
        return new_vector
    

    def solve(self,
              cost: torch.Tensor | np.ndarray[float] | None = None,
              **kwargs
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
        Converts a list of arcs to a one-hot encoded array.

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

        if any(node not in self.vertices for node in shortest_path_nodes):
            raise ValueError("Shortest path contains nodes that are not in the graph vertices.")

        # Create list of arcs form shortest path nodes
        shortest_path = [Graph.__sort(shortest_path_nodes[i],
                                              shortest_path_nodes[i + 1]
                                              )
                         for i in range(len(shortest_path_nodes) - 1)]
        # objective = sum(self.graph.edges[edge]['weight'] for edge in shortest_path)

        # Create a one-hot encoded array for the arcs
        num_arcs = len(self.arcs)
        arc_indices = [self.arcs.index(arc) for arc in shortest_path]
        one_hot_vector = np.zeros(num_arcs, dtype=np.float32)
        one_hot_vector[arc_indices] = 1.0
        objective = one_hot_vector @ self.cost

        return one_hot_vector, objective

    def visualize(self,
                  colored_edges: np.ndarray | None = None,
                  dashed_edges: np.ndarray | None = None,
                  figsize: tuple[int, int] = (6,5)
                  ) -> None:
        """
        Very simple visualization of the graph without edge annotations.
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Add edges from your list of arcs
        G.add_edges_from(self.arcs)

        # Draw the graph
        plt.figure(figsize=figsize)
        nx.draw(G, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=12, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray')

        plt.title("Graph from Arcs")
        plt.show()
    
    def _getModel(self):

        return self.graph, self.cost

    def setObj(self, 
               c: np.ndarray[float] | list[float] | float | None,
               source: int = None,
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

        # Set source and target nodes
        self._set_source_target(source, target)

        # Check if cost is in the correct data format
        if isinstance(c, (list, np.ndarray)):
            cost = np.squeeze(c)
        else:
            raise TypeError(
                f"Expected cost to be ndarray or list, got {type(c)} instead."
            )
        # Check if cost is a 1D arrays
        if cost.ndim != 1:
            raise ValueError(
                f"Expected costs to be a 1D array, got {cost.ndim}D array instead."
            )
        # Check if the length of cost matches the number of arcs
        if len(cost) != len(self.arcs) and len(cost) != 1:
            raise ValueError(f"cost has length {len(c)}, expected {len(self.arcs)}")
        
        # Store cost attribute if all checks pass
        self.cost = cost

        # Add edges to the graph with the specified weights
        for i, arc in enumerate(self.arcs):
            u, v = arc
            w = cost[i] if len(cost) > 1 else cost[0]
            self.graph.add_edge(u, v, weight=w)
        
        # TODO: There is a functionality that len(cost) can be 1. 
        # Remove it: Update length check and w assignment for nx graph.
        # When it's removed, use the _to_1d_numpy method for conversion.
        # When using _to_1d_numpy, remove obsolete tests for setObj.
        pass

    def _set_source_target(self, 
                           source: int | None, 
                           target: int | None) -> None:
        """
        Sets the source and target nodes for the graph.

        ------------
        Parameters
        ------------
        source : int, optional
            The source node for the shortest path. Defaults to the first vertex.
        target : int, optional
            The target node for the shortest path. Defaults to the last vertex.
        ------------
        Raises
        ------------
        TypeError : If the source or target is not an integer or None.
        ValueError : If the source or target node is not in the graph vertices.
        ------------
        """
        # TODO: Change that target defaults to the last vertex instead of the largest value

        # Check that source and target are of the type int or None
        if not (isinstance(source, int) or source is None):
            raise TypeError(f"Expected source to be an integer or None, got {type(source)} instead.")
        if not (isinstance(target, int) or target is None):
            raise TypeError(f"Expected target to be an integer or None, got {type(target)} instead.")
        
        # Raise an error if source or target don't exist in the graph
        if source is not None and source not in self.vertices:
            raise ValueError(f"Source node {source} is not in the graph vertices.")
        if target is not None and target not in self.vertices:
            raise ValueError(f"Target node {target} is not in the graph vertices.")

        # Set source to the first vertex if not provided
        if source is None:
            self.source = self.vertices[0]
        else:
            self.source = source

        # Set target to the vertex with highest number if not provided
        if target is None:
            self.target = max(self.vertices)
        else:
            self.target = target
        
        pass

    @staticmethod
    def one_hot_to_arcs(model: 'Graph',
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
        # TODO: Make this a class method.
        # TODO: Rename to _one_hot_to_arcs for consistency with _arcs_one_hot.
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
