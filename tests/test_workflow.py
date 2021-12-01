import pytest
import networkx as nx

from optinet import workflow_verification as wv


@pytest.mark.parametrize("nodes, edges, expected", [
    (10, 20, 30),
    (5, 6, 11),
])
def test_good_beh(nodes, edges, expected):
    g_obj, nd_ed = wv.generate_a_basic_random_graph(nodes, edges)
    assert isinstance(g_obj, nx.classes.graph.Graph)
    assert nd_ed == nodes + edges


# Chek whether an error is raised when needed
@pytest.mark.parametrize("nodes, edges", [
    (4, 10),
    (5, 60),
])
def test_error(nodes, edges):
    with pytest.raises(ValueError):
        wv.generate_a_basic_random_graph(nodes, edges)
