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
    Lightweight shortest-path model built on top of a weighted `networkx.DiGraph`.

    This class is the stateful graph representation used directly in the model-layer
    tests and indirectly by `ShortestPathGrb`, which deep-copies a `Graph` instance
    and then mirrors its cost vector into a Gurobi objective. The ordering of
    `self.arcs` is therefore the central convention of the class: every cost vector,
    one-hot path vector, solution vector, and interdiction vector is interpreted in
    that exact arc order.

    Core attributes
    ---------------
    `arcs`
        Ordered list of graph arcs. This defines the index layout for optimization
        inputs and outputs.
    `vertices`
        Available node labels.
    `cost`
        Current edge-weight vector aligned with `arcs`.
    `graph`
        Internal `networkx.DiGraph` whose edge weights are kept synchronized with
        `cost`.
    `source`, `target`
        Terminal nodes for shortest-path solves.

    Main responsibilities
    ---------------------
    `__init__`
        Instantiates the graph from arcs, optional vertices, optional costs, and
        optional source/target terminals. If vertices are omitted, they are inferred
        as `0..max endpoint in arcs`. If costs are omitted, every arc receives unit
        weight. Initialization creates the `networkx` graph, pushes the initial
        objective through `setObj`, stores source and target defaults, and then
        initializes the `optModel` parent class.
    `setObj`
        Central state-update method for the class. It validates and stores a new
        1D cost vector, optionally updates `source` and `target`, and rewrites the
        weights on every edge in the backing `networkx` graph. Both `Graph.solve`
        and `ShortestPathGrb.setObj` rely on this method to keep model state and
        edge weights consistent.
    `solve`
        Computes a shortest path between `source` and `target` using
        `networkx.shortest_path(..., weight="weight", method="dijkstra")`. It
        returns a one-hot arc vector plus its objective value. If a temporary cost
        vector is passed through `c` or `cost`, the method swaps that objective in,
        solves, and restores the original stored cost in a `finally` block. It also
        supports batched 2D torch cost tensors through `_solve_tensor`, returning one
        solution and objective per row while still restoring the original model state
        after the batch is processed.
    `evaluate`
        Scores a candidate one-hot path vector without solving. It computes the path
        objective under the stored `cost`, or under `cost + interdictions` when an
        additive interdiction vector is provided. This is a pure scoring step: it
        does not mutate the graph objective or the `networkx` weights.

    Important helpers
    -----------------
    `__call__`
        Alias for `evaluate`, allowing the graph to be used as a callable scorer.
    `_to_1d_numpy`
        Normalizes list, NumPy, and torch inputs into validated 1D NumPy arrays that
        match the arc dimension.
    `_solve_tensor`
        Batched shortest-path helper used by `solve` for 2D torch cost tensors.
    `_arcs_one_hot`
        Converts a node path into the arc-aligned one-hot representation returned by
        `solve`.
    `one_hot_to_arcs`
        Decodes a one-hot arc vector back into the corresponding list of arcs.
    `_set_source_target`
        Validates and stores terminal nodes.
    `visualize`, `__deepcopy__`, `_getModel`, `num_edges`
        Utility methods for plotting, copying, compatibility with the parent model
        interface, and simple graph metadata access.
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
        Build a shortest-path graph and initialize its current objective.

        Parameters
        ----------
        arcs : list[tuple[int, int]]
            Ordered arc list for the graph. This ordering defines the meaning of every
            cost vector, one-hot path vector, and interdiction vector used by the
            class.
        vertices : np.ndarray[int] | list[int], optional
            Explicit node labels. If omitted, vertices are inferred as the contiguous
            range `0..max endpoint in arcs`.
        cost : np.ndarray[float] | list[float], optional
            Initial cost vector aligned with `arcs`. If omitted, the graph starts with
            unit cost on every arc.
        source : int, optional
            Source node for future solves. If omitted, the first stored vertex is used.
        target : int, optional
            Target node for future solves. If omitted, the largest stored vertex is
            used.

        Notes
        -----
        Initialization creates an empty `networkx.DiGraph`, adds the vertices, pushes
        the initial weights through `setObj`, stores source and target defaults, and
        then calls the `optModel` parent constructor.

        Raises
        ------
        TypeError
            If `source` or `target` is not an integer or `None`.
        ValueError
            If `cost` cannot be interpreted as a 1D vector of length `len(arcs)`, or
            if `source` / `target` is not present in `vertices`.
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
        Create an independent copy of the graph model.

        Parameters
        ----------
        memo : dict
            A dictionary to keep track of already copied objects.

        Returns
        -------
        Graph
            A new `Graph` with copied arcs, vertices, and costs. Source and target are
            re-derived from the copied graph's defaults because they are not passed
            explicitly into the constructor here.
        """

        new_instance = Graph(
            arcs=deepcopy(self.arcs, memo),
            vertices=deepcopy(self.vertices, memo),
            cost=deepcopy(self.cost, memo),
            source=self.source,
            target=self.target,
        )
        return new_instance
    
    def __call__(self,
                 path: np.ndarray[float],
                 interdictions: np.ndarray[float] | None = None
                 ) -> float:
        """
        Alias for :meth:`evaluate`.

        Parameters
        ----------
        path : np.ndarray[float] | torch.Tensor | list[float]
            One-hot or flow-style arc vector aligned with `self.arcs`.
        interdictions : np.ndarray[float] | None, optional
            Optional additive cost adjustment applied only for this evaluation. The
            stored graph objective is not modified.

        Returns
        -------
        float
            The path objective under the stored costs, or under the stored costs plus
            the provided interdictions.
        """

        return self.evaluate(path, interdictions)
    
    # TODO: Add equals method. Then update tests in, e.g., test_shortest_path_grb.
    
    def evaluate(self,
                 path: np.ndarray[float],
                 interdictions: np.ndarray[float] | None = None
                 ) -> float:
        """
        Score a provided path under the current graph objective.

        Parameters
        ----------
        path : np.ndarray[float] | torch.Tensor | list[float]
            One-dimensional arc vector aligned with `self.arcs`. The method accepts
            NumPy arrays, torch tensors, and Python lists, and normalizes them through
            `_to_1d_numpy`.
        interdictions : np.ndarray[float] | torch.Tensor | list[float] | None, optional
            Optional additive arc-cost vector of the same length as `path`. If
            provided, the returned objective is `path @ (self.cost + interdictions)`.
            The graph's stored objective and edge weights are not updated.

        Returns
        -------
        float
            Scalar objective value of the provided path.

        Raises
        ------
        TypeError
            If `path` or `interdictions` is not a supported vector type.
        ValueError
            If `path` or `interdictions` cannot be reduced to a 1D vector of length
            `len(self.arcs)`.
        """

        # Convert provided path to numpy array
        new_path = self._to_1d_numpy(path)

        # Return the objective value if no interdictions are provided
        if interdictions is None:
            return new_path @ self.cost
        
        # Convert provided interdictions to numpy array
        new_interdictions = self._to_1d_numpy(interdictions)

        return new_path @ (self.cost + new_interdictions)
    # TODO: Evaluate method should also be able to handle 2D torch.tensors just like solve.


    def _to_1d_numpy(
            self, 
            vector: np.ndarray[float] | torch.Tensor | list[float]
        ) -> np.ndarray[float]:
        """
        Normalize a supported vector input into a validated 1D NumPy array.

        The method accepts Python lists, NumPy arrays, and torch tensors. Inputs are
        squeezed before validation, so shapes such as `(1, n)` are accepted if they
        reduce to a single 1D vector. The final vector must match the arc dimension of
        the graph.

        Parameters
        ----------
        vector : np.ndarray[float] | torch.Tensor | list[float]
            Candidate vector to normalize.

        Returns
        -------
        np.ndarray[float]
            Copy of the provided data as a 1D NumPy array.

        Raises
        ------
        TypeError
            If `vector` is not a list, NumPy array, or torch tensor.
        ValueError
            If the squeezed result is not 1D, or if its length differs from
            `len(self.arcs)`.
        """
        # Convert vector to numpy array if it's a torch tensor or list.
        # Raise error if it's not one of the expected types.
        if isinstance(vector, torch.Tensor):
            new_vector = vector.detach().cpu().numpy().squeeze()
        elif isinstance(vector, (list, np.ndarray)):
            new_vector = np.array(np.squeeze(vector), copy=True)
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
              c: torch.Tensor | np.ndarray[float] | list[float] | None = None,
              **kwargs
              ) -> Tuple[np.ndarray, float]:
        """
        Solve the current shortest-path problem.

        The solve is performed with `networkx.shortest_path` using edge attribute
        `"weight"` and `method="dijkstra"`. By default the method uses the graph's
        stored `self.cost`. A temporary cost vector can be supplied either as `c` or
        as keyword argument `cost`; when both are supplied, `c` takes precedence and
        `kwargs["cost"]` is ignored.

        If a temporary 1D cost vector is provided, the method updates the graph
        objective via `setObj`, solves, and then restores the original stored costs in
        a `finally` block. If a temporary 2D torch tensor is provided, the method
        delegates to `_solve_tensor` and returns one solution and one objective per
        row.

        Parameters
        ----------
        c : torch.Tensor | np.ndarray[float] | list[float] | None, optional
            Temporary cost input. Supported cases are:
            - 1D list / NumPy array / torch tensor: solve one instance and return a
              NumPy solution vector plus scalar objective.
            - 2D torch tensor with shape `(batch_size, len(self.arcs))`: solve a
              batch of instances and return torch tensors for both solutions and
              objectives.
        **kwargs
            Optional keyword arguments. The method recognizes `cost` as an alternate
            name for `c`.

        Returns
        -------
        tuple[np.ndarray, float] | tuple[torch.Tensor, torch.Tensor]
            For a single solve, returns `(solution, objective)` where `solution` is a
            one-hot NumPy vector aligned with `self.arcs` and `objective` is the path
            cost. For batched 2D torch input, returns `(solutions, objectives)` as
            torch tensors, with one row and one objective per input row.

        Raises
        ------
        ValueError
            If a provided temporary cost vector has the wrong shape.
        networkx.NetworkXNoPath
            If no path exists between the current source and target.
        """
        new_cost = False
        # Cover case if new cost is provided
        if c is not None or "cost" in kwargs:
            new_cost = True
            if c is not None:
                # Case: c is not None, "cost" can be in kwargs but ignored
                cost = c.copy() 
            else:
                # Case: c is None, "cost" in kwargs
                cost = kwargs.pop("cost")
        
            # Check if cost is a 2D torch tensor
            if isinstance(cost, torch.Tensor) and cost.ndim == 2:
                return self._solve_tensor(cost, self.source, self.target)

            # All remaining options are 1D: list, numpy array, torch tensor 
            # and will be checked in setObj

            # Store original cost then update objective
            original_cost = self.cost.copy()
            self.setObj(cost)

        try:
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
        finally:
            # Restore cost if it was provided
            if new_cost:
                self.setObj(original_cost)

        return shortest_path, objective
    
    def _solve_tensor(self, 
                      costs: torch.Tensor,
                      source: int,
                      target: int) -> None:
        """
        Solve a batch of shortest-path instances from a 2D torch cost tensor.

        Parameters
        ----------
        costs : torch.Tensor
            Tensor with shape `(batch_size, len(self.arcs))`. Each row is treated as a
            temporary cost vector for one shortest-path solve.
        source : int
            Source node to use for every batch item.
        target : int
            Target node to use for every batch item.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Batched solution tensor of shape `(batch_size, len(self.arcs))` and
            batched objective tensor of shape `(batch_size,)`.

        Notes
        -----
        The method temporarily overwrites the graph objective row by row, computes the
        corresponding shortest path, and restores the original cost/source/target in a
        `finally` block.

        Raises
        ------
        ValueError
            If `costs` is not a 2D tensor with one column per arc.
        """
        
        # Ensure costs is a 2D tensor of appropriate shape
        costs_arr = costs.detach().cpu().numpy()
        if costs_arr.ndim != 2:
            raise ValueError(
                f"Expected costs to be a 2D tensor, got {costs_arr.ndim}D tensor instead.")
        if costs_arr.shape[1] != len(self.arcs):
            raise ValueError(
                f"Expected costs to have {len(self.arcs)} columns, " + 
                f"got {costs_arr.shape[1]} columns instead.")

        original_cost = self.cost.copy()
        original_source = self.source
        original_target = self.target

        solutions = np.zeros((costs_arr.shape[0], len(self.arcs)), dtype=np.float32)
        objectives = np.empty(costs_arr.shape[0], dtype=costs_arr.dtype)

        try:
            for i, cost in enumerate(costs_arr):
                self.setObj(cost, source=source, target=target)
                shortest_path_nodes = nx.shortest_path(
                    self.graph,
                    source=self.source,
                    target=self.target,
                    weight='weight',
                    method='dijkstra'
                )
                solutions[i], objectives[i] = self._arcs_one_hot(shortest_path_nodes)
        finally:
            self.setObj(original_cost, source=original_source, target=original_target)

        return torch.from_numpy(solutions), torch.from_numpy(objectives)
    
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
        Convert a node-path representation into the graph's one-hot arc format.

        Parameters
        ----------
        shortest_path_nodes : list[int]
            Ordered node sequence returned by a shortest-path routine.

        Returns
        -------
        tuple[np.ndarray[float], float]
            One-hot arc vector aligned with `self.arcs`, and the corresponding
            objective computed against `self.cost`.

        Notes
        -----
        Consecutive node pairs are normalized through `__sort` before lookup in
        `self.arcs`, so this helper assumes the stored arc list is compatible with
        that normalization.

        Raises
        ------
        ValueError
            If any node is not present in `self.vertices`, or if a derived arc cannot
            be found in `self.arcs`.
        """

        if any(node not in self.vertices for node in shortest_path_nodes):
            raise ValueError("Shortest path contains nodes that are not in the graph vertices.")

        # Create list of arcs form shortest path nodes
        # shortest_path = [Graph.__sort(shortest_path_nodes[i],
        #                                       shortest_path_nodes[i + 1]
        #                                       )
        #                  for i in range(len(shortest_path_nodes) - 1)]
        shortest_path = [(shortest_path_nodes[i],
                          shortest_path_nodes[i + 1]
                          ) for i in range(len(shortest_path_nodes) - 1)]
        # objective = sum(self.graph.edges[edge]['weight'] for edge in shortest_path)

        # Create a one-hot encoded array for the arcs
        num_arcs = len(self.arcs)
        arc_indices = [self.arcs.index(arc) for arc in shortest_path] # raises ValueError if arc not in list
        one_hot_vector = np.zeros(num_arcs, dtype=np.float32)
        one_hot_vector[arc_indices] = 1.0
        objective = one_hot_vector @ self.cost

        return one_hot_vector, objective

    def visualize(self,
                  colored_edges: np.ndarray | None = None,
                  dashed_edges: np.ndarray | None = None,
                  figsize: tuple[int, int] = (6,5),
                  *,
                  ax: plt.Axes | None = None,
                  title: str | None = "Graph from Arcs",
                  layout_seed: int | None = 7,
                  width: float = 1.5,
                  **kwargs
                  ) -> None:
        """
        Display a simple plot of the current graph structure.

        Parameters
        ----------
        colored_edges : np.ndarray | None, optional
            A one-hot encoded vector representing the edges to color red. If None,
            no edges are colored.
        dashed_edges : np.ndarray | None, optional
            A one-hot encoded vector representing the edges to draw as dashed. If
            None, all edges use a solid line style.
        figsize : tuple[int, int], optional
            Matplotlib figure size.
        ax : plt.Axes | None, optional
            Matplotlib axes to draw on. If None, a new figure is created and shown.
        title : str | None, optional
            Plot title. If None, no title is set.
        layout_seed : int | None, optional
            Seed passed to ``networkx.spring_layout`` for stable node placement.
        width : float, optional
            Width of the graph edges.
        **kwargs
            Additional keyword arguments forwarded to
            ``networkx.draw_networkx_edges`` for the base edge drawing.
        """
        # Validate optional one-hot masks before creating matplotlib state.
        colored_arcs = None
        if colored_edges is not None:
            colored_arcs = set(Graph.one_hot_to_arcs(
                self,
                self._to_1d_numpy(colored_edges),
            ))

        dashed_arcs = None
        if dashed_edges is not None:
            dashed_arcs = set(Graph.one_hot_to_arcs(
                self,
                self._to_1d_numpy(dashed_edges),
            ))

        # Draw using the stored graph so isolated vertices are preserved.
        pos = nx.spring_layout(self.graph, seed=layout_seed)

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
            should_show = True
        else:
            plt.sca(ax)
            should_show = False
        ax.set_axis_off()

        edge_kwargs = {
            "edge_color": "gray",
            "width": width,
            "arrows": True,
            "arrowsize": 20,
        }
        edge_kwargs.update(kwargs)

        edge_artists = nx.draw_networkx_edges(
            self.graph,
            pos,
            ax=ax,
            **edge_kwargs,
        )
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            ax=ax,
            node_color="lightblue",
            node_size=500,
        )
        nx.draw_networkx_labels(
            self.graph,
            pos,
            ax=ax,
            font_size=12,
            font_weight="bold",
        )

        # Match each drawn edge artist to the corresponding graph edge.
        edge_list = list(self.graph.edges())
        if colored_arcs is not None:
            base_color = edge_kwargs["edge_color"]
            for patch, edge in zip(edge_artists, edge_list):
                patch.set_color("red" if edge in colored_arcs else base_color)

        if dashed_arcs is not None:
            for patch, edge in zip(edge_artists, edge_list):
                patch.set_linestyle(
                    "dashed" if edge in dashed_arcs else "solid"
                )

        if title is not None:
            ax.set_title(title)

        if should_show:
            plt.show()
        pass
    
    def _getModel(self):
        """
        Return the lightweight model representation expected by the parent interface.

        Returns
        -------
        tuple[nx.DiGraph, np.ndarray]
            The underlying `networkx` graph together with the current cost vector.
        """

        return self.graph, self.cost

    def setObj(self, 
               c: np.ndarray[float] | list[float] | float | None,
               source: int = None,
               target: int = None
               ) -> None:
        """
        Update the graph objective and synchronize the `networkx` edge weights.

        This is the mutating objective setter for the class. It optionally updates
        `source` and `target`, validates the provided cost vector through
        `_to_1d_numpy`, stores the result in `self.cost`, and rewrites the weight on
        every stored arc in `self.graph`.

        Parameters
        ----------
        c : np.ndarray[float] | torch.Tensor | list[float]
            New 1D cost vector aligned with `self.arcs`. Inputs are squeezed and then
            validated by `_to_1d_numpy`.
        source : int, optional
            Optional replacement source node. If `None`, the current source is
            preserved after initialization.
        target : int, optional
            Optional replacement target node. If `None`, the current target is
            preserved after initialization.

        Raises
        ------
        TypeError
            If `c` is not a supported vector type, or if `source` / `target` is not
            an integer or `None`.
        ValueError
            If `c` cannot be reduced to a 1D vector of length `len(self.arcs)`, or if
            `source` / `target` is not present in the graph.
        """

        # Preserve existing terminals when only the objective changes. During
        # construction these attributes do not exist yet, so None still means
        # "use the graph defaults."
        resolved_source = (
            self.source
            if source is None and hasattr(self, "source")
            else source
        )
        resolved_target = (
            self.target
            if target is None and hasattr(self, "target")
            else target
        )

        # Set source and target nodes
        self._set_source_target(resolved_source, resolved_target)

        # Convert to 1D numpy array
        cost = self._to_1d_numpy(c)
        
        # Store cost attribute if all checks pass
        self.cost = cost

        # Add edges to the graph with the specified weights
        for i, arc in enumerate(self.arcs):
            u, v = arc
            w = cost[i] if len(cost) > 1 else cost[0]
            self.graph.add_edge(u, v, weight=w)
        pass

    def _set_source_target(self, 
                           source: int | None, 
                           target: int | None) -> None:
        """
        Validate and store the terminals used by `solve`.

        Parameters
        ----------
        source : int | None
            Source node. If `None`, the first stored vertex is used.
        target : int | None
            Target node. If `None`, the largest stored vertex is used.

        Raises
        ------
        TypeError
            If `source` or `target` is not an integer or `None`.
        ValueError
            If an explicit `source` or `target` does not belong to `self.vertices`.
        """
        # TODO: Change that target defaults to the last vertex instead of the largest value

        integer_types = (int, np.integer)

        # Check that source and target are of the type int or None
        if not (isinstance(source, integer_types) or source is None):
            raise TypeError(f"Expected source to be an integer or None, got {type(source)} instead.")
        if not (isinstance(target, integer_types) or target is None):
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

        if isinstance(self.source, np.integer):
            self.source = int(self.source)
        if isinstance(self.target, np.integer):
            self.target = int(self.target)
        
        pass

    @staticmethod
    def one_hot_to_arcs(model: 'Graph',
                         one_hot_vector: np.ndarray[float]
                         ) -> list[tuple[int, int]]:
        """
        Decode a one-hot arc vector into the corresponding arc list.

        Parameters
        ----------
        model : Graph
            Graph instance whose `arcs` ordering defines the decoding.
        one_hot_vector : np.ndarray[float]
            Arc indicator vector aligned with `model.arcs`. Every strictly positive
            entry is interpreted as "arc selected".

        Returns
        -------
        list[tuple[int, int]]
            Subsequence of `model.arcs` selected by the positive entries.
        """
        # TODO: Make this a class method.
        # TODO: Rename to _one_hot_to_arcs for consistency with _arcs_one_hot.
        return [arc for arc in model.arcs if one_hot_vector[model.arcs.index(arc)] > 0]

    @property
    def num_edges(self) -> int:
        """
        Number of stored arcs in the graph.
        """
        return len(self.arcs)
