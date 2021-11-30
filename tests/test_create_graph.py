import pytest
import pandas as pd

from optinet.create_graph import CitiesNodes

df = pd.read_csv("tests/test_data/Cities.csv", sep=";", decimal=",")


@pytest.mark.parametrize("node_attrs", [df, df.to_dict("index")])
def test_init_and_lengths(node_attrs):
    graph = CitiesNodes(node_attrs=node_attrs)
    graph.add_edges_from([[1, 2], [0, 1], [1, 3]])
    graph.set_edge_lengths()
    assert 250 <= graph.edges[0, 1]["length"] <= 260            # Distance between Warsaw and Krakow
    assert 250 <= graph.calculate_edge_lengths([0, 1]) <= 260   # should be in that range


@pytest.mark.parametrize("edges, error", [
    (None, ValueError),
    ([[1, 2, 3], [1, 2, 3]], ValueError),       # Edge with 3 nodes is impossible (not a multigraph)
])
def test_length_errors(edges, error):
    graph = CitiesNodes(node_attrs=df)
    with pytest.raises(error):
        graph.calculate_edge_lengths(edges)


def test_initialize_graph():
    node_attrs = "This is certainly not a dictionary or a pandas DataFrame"
    with pytest.raises(TypeError):
        CitiesNodes(node_attrs=node_attrs)


def test_set_edge_lengths_error():
    graph = CitiesNodes(node_attrs=df)
    with pytest.raises(RuntimeError):
        graph.set_edge_lengths()                # There are no edges in the graph yet
