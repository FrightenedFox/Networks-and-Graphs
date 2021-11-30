import pytest
import pandas as pd
import networkx as nx

from optinet.create_graph import CitiesNodes

df = pd.read_csv("../Data/Cities.csv", sep=";", decimal=",")[:20]


def test_graph_generation():
    # TODO: check what happens if there is only one node, a lot of nodes etc.
    graph = CitiesNodes(node_attrs=df)
    print(graph.calculate_edge_length([1, 0]))
    # TODO: somehow test length generation


# @pytest.mark.parametrize("graph, error", [
#     (adjacency_matrix, TypeError),                      # not a graph
#     (nx.DiGraph(), NotImplementedError),                # directed graph
#     (nx.MultiGraph(), NotImplementedError),             # multigraph
#     (nx.MultiDiGraph(), NotImplementedError),           # directed multigraph
# ])
# def test_graph_errors(graph, error):
#     with pytest.raises(error):
#         mutate.from_graph(graph, prob=10)

if __name__ == "__main__":
    test_graph_generation()
