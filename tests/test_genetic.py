import pytest
import numpy as np
import networkx as nx

from optinet.genetic import mutate

adjacency_matrix = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
])


def test_from_matrix():
    mut_matrix = mutate.from_adjacency_matrix(adjacency_matrix)[0]
    assert mut_matrix.ndim == 2
    assert mut_matrix.shape[0] == mut_matrix.shape[1]
    assert np.array_equal(mut_matrix, mut_matrix.T)


def test_from_graph():
    graph = nx.from_numpy_array(adjacency_matrix)
    mut_graph = mutate.from_graph(graph)
    assert isinstance(mut_graph, nx.Graph)
    assert len(mut_graph.nodes) == adjacency_matrix.shape[0]


@pytest.mark.parametrize("matrix, prob, error", [
    (np.eye(2), 1000, ValueError),                          # prob > 1
    (np.eye(2), 1.2, ValueError),                           # prob > 1
    (([[0, 1], [1, 0]]), 10, TypeError),                    # not a numpy.array
    (np.array([0, 1, 0, 1]), 10, ValueError),               # not two-dimensional
    (np.array([[0, 1], [1, 0], [1, 1]]), 10, ValueError),   # not two-dimensional
    (np.array([[0, 2], [2, 0]]), 1, NotImplementedError),   # hypergraph
])
def test_adjacency_matrix_errors(matrix, prob, error):
    with pytest.raises(error):
        mutate.from_adjacency_matrix(matrix, prob=prob)


@pytest.mark.parametrize("graph, error", [
    (adjacency_matrix, TypeError),      # not a graph
    (nx.DiGraph(), TypeError),          # directed graph
    (nx.MultiGraph(), TypeError),       # multigraph
    (nx.MultiDiGraph(), TypeError),     # directed multigraph
])
def test_graph_errors(graph, error):
    with pytest.raises(error):
        mutate.from_graph(graph, prob=10)
