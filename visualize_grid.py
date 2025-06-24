import networkx as nx
import matplotlib.pyplot as plt

def visualize_grid(
    m: int,
    n: int,
    c: list[float] | tuple[float, ...],
    s: tuple[int, int],
    t: tuple[int, int],
    path: list[tuple[tuple[int,int],tuple[int,int]]] | None = None,
    special_edges: list[tuple[tuple[int,int],tuple[int,int]]] | None = None
) -> None:
    """
    Visualize an m×n grid with edge‐weights, start/end labels, 
    optional highlighted path, and optional dashed edges.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    c : sequence of float, length m*(n-1) + (m-1)*n
        Edge‐weights, first all horizontal edges (left→right, row by row),
        then all vertical edges (top→bottom, column by column).
    s, t : (i, j)
        Start and end node coordinates: 0 ≤ i < m, 0 ≤ j < n.
    path : array of 0-1 values where 1-values are highlighted in color, optional
        Edges to highlight in color.
    special_edges : list of edges, optional
        Edges to draw with dashed style.
    """
    # Build the grid graph
    G = nx.grid_2d_graph(m, n)
    pos = { (i,j):(j, -i) for i,j in G.nodes() }

    # Build ordered edge lists
    horiz = [((i,j),(i,j+1)) for i in range(m)    for j in range(n-1)]
    vert  = [((i,j),(i+1,j)) for j in range(n)    for i in range(m-1)]
    all_edges = horiz + vert

    if len(c) != len(all_edges):
        raise ValueError(f"c has length {len(c)}, expected {len(all_edges)}")

    # Assign weights
    for edge, w in zip(all_edges, c):
        u, v = edge
        G.edges[u, v]['weight'] = w

    # Draw base grid
    plt.figure(figsize=(n, m))
    nx.draw(
        G, pos,
        node_size=300,
        node_color='lightgray',
        edge_color='black',
        with_labels=False
    )

    # Draw weight labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Draw special (dashed) edges
    if special_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=special_edges,
            style='dashed',
            width=1.5
        )

    # Draw highlighted path
    if len(path) > 1:
        if len(path) != len(all_edges):
            raise ValueError(
                f"‘path’ must be length {len(all_edges)}, got {len(path)}"
            )
        # pick only those edges where path[i] == 1
        path_edges = [edge for edge, flag in zip(all_edges, path) if flag]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=path_edges,
            edge_color='red',
            width=2.5
        )

    # Label start and end
    labels = {node: '' for node in G.nodes()}
    labels[s] = 's'
    labels[t] = 't'
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='blue', font_size=12)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
