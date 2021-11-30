import pytest
import pandas as pd

from optinet.create_graph import CitiesNodes

df = pd.read_csv("tests/test_data/Cities.csv", sep=";", decimal=",")


@pytest.mark.parametrize("node_attrs", [df, df.to_dict("index")])
def test_init_and_lengths(node_attrs):
    graph = CitiesNodes(node_attrs=node_attrs)
    assert graph.multi_incidence_matrix is None
    graph.add_edges_from([[2, 1], [0, 1], [1, 3], [16, 0]])
    assert graph.multi_incidence_matrix.shape == (4, 24)        # type: ignore[attr-defined]
    graph.set_edge_lengths()
    assert 250 <= graph.edges[0, 1]["length"] <= 260            # Distance between Warsaw and Krakow
    assert 250 <= graph.calculate_edge_lengths([0, 1]) <= 260   # should be in that range
    assert 920 <= graph.total_length <= 960


@pytest.mark.parametrize("edges, error", [
    (None, ValueError),
    ([[1, 2, 3], [1, 2, 3]], ValueError),       # Edge with 3 nodes is impossible (not a multigraph)
])
def test_length_errors(edges, error):
    graph = CitiesNodes(node_attrs=df)
    with pytest.raises(error):
        graph.calculate_edge_lengths(edges)


# noinspection PyTypeChecker
def test_initialize_graph_error():
    node_attrs = "This is certainly not a dictionary or pandas DataFrame"
    with pytest.raises(TypeError):
        CitiesNodes(node_attrs=node_attrs)      # Not a dictionary or pandas DataFrame


def test_set_edge_lengths_error():
    graph = CitiesNodes(node_attrs=df)
    with pytest.raises(RuntimeError):
        graph.set_edge_lengths()                # There are no edges in the graph yet
    assert graph.total_length == 0


if __name__ == "__main__":
    test_init_and_lengths(df)
