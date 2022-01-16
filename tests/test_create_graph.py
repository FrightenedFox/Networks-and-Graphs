import networkx as nx
import pytest
import pandas as pd
import matplotlib.pyplot as plt

from optinet.create_graph import CitiesNodes

# df = pd.read_csv("/home/vitalii/git_projects/Networks-and-Graphs/tests/test_data/Cities.csv", sep=";", decimal=",")
# mim_df = pd.read_csv("/home/vitalii/git_projects/Networks-and-Graphs/tests/test_data/multi_incidence_matrix.csv", index_col=[0, 1])
df = pd.read_csv("C:/Users/vital/git_projects/Networks-and-Graphs/tests/test_data/Cities.csv", sep=";", decimal=",")
mim_df = pd.read_csv("C:/Users/vital/git_projects/Networks-and-Graphs/tests/test_data/multi_incidence_matrix.csv", index_col=[0, 1])
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

    graph.update(nx.complete_graph(n=len(graph.nodes)))
    graph.optimise(
        n_generations=100, 
        population_survival_size=12, 
        redundancy_level=3.5, 
        redundancy_sigma=0.2, 
        reproduction_mutation_prob=0.001)
    print(f"Total length of the graph:\t{graph.total_length:.3f} km")
    print(f"Edge connectivity:\t\t{nx.edge_connectivity(graph)}")
    print(f"Node connectivity:\t\t{nx.node_connectivity(graph)}")
    print(f"Average node connectivity:\t{nx.average_node_connectivity(graph):.3f}")
    nx.draw(graph, 
            pos=list(zip(df.Longitude, df.Latitude)), 
            with_labels=False, 
            node_color="cyan", 
            node_size=df.Population / 1e3, 
            labels=nx.get_node_attributes(graph,"City"),\
            font_color="k")
    plt.show()


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


if __name__ == '__main__':
    test_lengths(CitiesNodes(df))
