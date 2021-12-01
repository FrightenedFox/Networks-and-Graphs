import pytest
import pandas as pd

from optinet.create_graph import CitiesNodes

df = pd.read_csv("tests/test_data/Cities.csv", sep=";", decimal=",")
mim_df = pd.read_csv("tests/test_data/multi_incidence_matrix.csv", index_col=[0, 1])
node_list1 = [[2, 1], [0, 1], [1, 3], [16, 0]]
node_list2 = [[4, 2], [10, 1]]


@pytest.mark.parametrize("node_attrs", [df, df.to_dict("index")])
def test_init(node_attrs):
    graph = CitiesNodes(node_attrs=node_attrs)
    assert len(graph.nodes) == len(node_attrs)


def test_lengths(graph):
    with pytest.raises(RuntimeWarning):
        graph.set_edge_lengths()                # There are no edges in the graph yet
    assert graph.total_length == 0

    graph.add_edges_from(node_list1)
    graph.set_edge_lengths()
    assert 250 <= graph.edges[0, 1]["length"] <= 260            # Distance between Warsaw and Krakow
    assert 250 <= graph.calculate_edge_lengths([0, 1]) <= 260   # should be in that range
    assert 920 <= graph.total_length <= 960


def test_multi_incidence_matrix(graph):
    assert graph.multi_incidence_matrix is None
    graph.multi_incidence_matrix = None
    assert graph.multi_incidence_matrix is None
    graph.add_edges_from(node_list1)

    assert graph.multi_incidence_matrix.shape == (4, 24)    # type: ignore[attr-defined, union-attr]
    graph.multi_incidence_matrix = mim_df
    assert graph.multi_incidence_matrix.sum().sum() == 48   # 0.5 * (4 rows) * (24 cols)

    graph.add_edges_from(node_list2)
    assert graph.multi_incidence_matrix.shape == (6, 24)    # type: ignore[attr-defined]
    assert graph.multi_incidence_matrix.sum().sum() == 96   # 0.5 * (4 rows) * (24 cols) + 1.0 * (2 rows) * (24 cols)

    graph.remove_edges_from(node_list2)
    assert graph.multi_incidence_matrix.shape == (4, 24)    # type: ignore[attr-defined]
    assert graph.multi_incidence_matrix.sum().sum() == 48   # 0.5 * (4 rows) * (24 cols)


@pytest.mark.parametrize("value", [
    None,                               # Can not be None when there are edges in the graph
    mim_df.iloc[:2],                    # The shape of the DataFrame is not compatible with graph
    mim_df.reset_index(drop=True),      # Index doesn't represent actual edges of the graph
])
def test_multi_incidence_matrix_errors(value, graph):
    graph.add_edges_from(node_list1)
    with pytest.raises(ValueError):
        graph.multi_incidence_matrix = value


@pytest.mark.parametrize("edges, error", [
    (None, ValueError),
    ([[1, 2, 3], [1, 2, 3]], ValueError),       # Edge with 3 nodes is impossible (not a multigraph)
])
def test_length_errors(edges, error, graph):
    with pytest.raises(error):
        graph.calculate_edge_lengths(edges)


# noinspection PyTypeChecker
def test_initialize_graph_error():
    node_attrs = "This is certainly not a dictionary or pandas DataFrame"
    with pytest.raises(TypeError):
        CitiesNodes(node_attrs=node_attrs)      # Not a dictionary or pandas DataFrame
