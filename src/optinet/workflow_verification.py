import networkx as nx


def generate_a_basic_random_graph(n_nodes, n_edges):
    if n_edges > (n_nodes - 1) * n_nodes / 2:
        raise ValueError(f"Given number of edges ({n_edges}) is larger than the number "
                         f"of edges in a complete graph with {n_nodes} nodes.")
    g = nx.gnm_random_graph(n_nodes, n_edges)
    return g, len(g.edges) + len(g.nodes)
