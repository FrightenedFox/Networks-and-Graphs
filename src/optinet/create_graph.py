from typing import Union, Dict, Any, Optional

import numpy as np
import networkx as nx
import pandas as pd
from geopy.distance import distance

from optinet.genetic import mutate


class CitiesNodes(nx.Graph):

    def __init__(self, node_attrs: Union[Dict[int, Dict[str, Any]], pd.DataFrame] = None, **kwargs):
        """Creates networkx Graph with node attributes if passed.

        Parameters
        ----------
        node_attrs : (optional)
            Dictionary or Pandas DataFrame with information about the nodes and attributes to be added to the graph.
        kwargs :
            networkx.Graph arguments.
        """
        super().__init__(**kwargs)

        self.node_attrs = node_attrs
        self.min_total_length: float = 0

        self._total_length: float = 0
        self._multi_incidence_matrix: Optional[pd.DataFrame] = None

        if isinstance(node_attrs, pd.DataFrame):
            self.add_nodes_from(node_attrs.index)
            nx.set_node_attributes(self, node_attrs.to_dict("index"))
        elif isinstance(node_attrs, dict):
            self.add_nodes_from(node_attrs.keys())
            nx.set_node_attributes(self, node_attrs)
        elif node_attrs is not None:
            raise TypeError(f"node_attrs of type dictionary or pandas.DataFrame is expected, "
                            f"{type(node_attrs)} is obtained instead.")

    def set_min_total_length(self):
        self.update(nx.complete_graph(n=len(self.nodes)))
        self.set_edge_lengths()
        min_sp_tree_graph = nx.minimum_spanning_tree(self, weight="length")
        self.clear_edges()
        self.update(min_sp_tree_graph)
        self.min_total_length = self.total_length
        self.clear_edges()
        return self.min_total_length

    def calculate_edge_lengths(self, edges):
        """Calculate length for each edge in edges.

        Parameters
        ----------
        edges : numpy.typing.ArrayLike
            One or more edges to calculate their length.

        Returns
        -------
        numpy.typing.ArrayLike
            The lengths of the given edges.
        """
        longitudes = pd.Series(nx.get_node_attributes(self, "Longitude"))
        latitude = pd.Series(nx.get_node_attributes(self, "Latitude"))
        coords = list(zip(latitude, longitudes))

        def edge_length(n1, n2):
            return distance(coords[n1], coords[n2]).km

        if not edges:
            raise ValueError("`edges` can not be None")
        elif np.shape(edges[0]):
            # There are a lot of edges
            if np.shape(edges)[1] != 2:
                raise ValueError(f"The second dimension of `edges` must be equal 2. Current shape: {np.shape(edges)}")

            lengths = np.empty(len(edges), dtype=float)
            for i, edge in enumerate(edges):
                lengths[i] = edge_length(*edge)
            return lengths
        else:
            # There is only one edge
            return edge_length(*edges)

    def set_edge_lengths(self):
        """Set the length for each edge."""
        if not self.edges:
            raise RuntimeWarning("The graph has no edges, no edge lengths were set.")
        else:
            lengths = self.calculate_edge_lengths(list(self.edges))
            nx.set_edge_attributes(self, dict(zip(self.edges, lengths)), name="length")

    @property
    def total_length(self) -> float:
        """Total length of all edges."""
        if self.edges:
            self.set_edge_lengths()
        self._total_length = sum(nx.get_edge_attributes(self, "length").values())
        return self._total_length

    def evaluate(self, population, length_coef=3, disconn_coef=2, conn_coef=1):
        self.set_min_total_length()
        scores = np.zeros(population.shape[0])
        for i, individual in enumerate(population):
            new_graph = CitiesNodes(self.node_attrs)
            new_graph.update(nx.from_numpy_array(individual))
            length_score = length_coef * self.min_total_length / new_graph.total_length
            disconn_score = disconn_coef * int(nx.is_connected(self))
            conn_score = conn_coef * nx.average_node_connectivity(self)
            scores[i] = length_score + disconn_score + conn_score
        return scores

    def optimise(self, n_mutations=3, population_size=10, redundancy_level=2):
        adj_mat = nx.to_numpy_array(self, dtype=int)
        st_edge_prob = 2 / len(self.nodes)  # probability of the edge in the spanning tree of this graph
        init_prob = redundancy_level * st_edge_prob
        init_prob = 0.5 if init_prob >= 1 else init_prob
        population = mutate.from_adjacency_matrix(
            adjacency_matrix=np.zeros(adj_mat.shape),
            prob=init_prob,
            n=population_size,
        )
        print(self.evaluate(population))

        pass
