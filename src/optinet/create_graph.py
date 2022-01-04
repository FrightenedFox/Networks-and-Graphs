from typing import Union, Dict, Any, Optional

import numpy as np
import networkx as nx
import pandas as pd
from geopy.distance import distance
import matplotlib.pyplot as plt

from optinet import mutate
from optinet import cross_section


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


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

    def evaluate(self, population, redundancy_level=1, redundancy_sigma=0.5):
        self.set_min_total_length()
        scores = np.zeros(population.shape[0])
        best_con_score, best_length_score, best_score, best_discon = 0, 0, 0, False
        for i, individual in enumerate(population):
            new_graph = CitiesNodes(self.node_attrs)
            new_graph.update(nx.from_numpy_array(individual))
            length_score = self.min_total_length / new_graph.total_length * redundancy_level
            disconn_score = int(nx.is_connected(new_graph))
            x = nx.average_node_connectivity(new_graph)
            conn_score = gaussian(x, redundancy_level, redundancy_sigma)
            scores[i] = length_score + disconn_score + conn_score
            if scores[i] > best_score:
                best_score = scores[i]
                best_con_score = conn_score
                best_length_score = length_score
                best_discon = nx.is_connected(new_graph)
        print(f"{best_length_score=} -- {best_con_score=} -- is connected? {best_discon}")
        return scores, np.argsort(scores)

    def optimise(self, n_generations=100, population_survival_size=12, redundancy_level=2, redundancy_sigma=0.1, reproduction_mutation_prob=0.001):
        initial_reproduction_mutation_prob = reproduction_mutation_prob
        adj_mat = nx.to_numpy_array(self, dtype=int)
        st_edge_prob = 2 / len(self.nodes)  # probability of the edge in the spanning tree of this graph
        init_prob = redundancy_level * st_edge_prob
        init_prob = 0.5 if init_prob >= 1 else init_prob
        population = mutate.from_adjacency_matrix(
            adjacency_matrix=np.zeros(adj_mat.shape),
            prob=init_prob,
            n=population_survival_size ** 2,
        )
        previous_best, n_repeats = 0, 0
        for generation in range(n_generations):
            scores, places = self.evaluate(population, redundancy_level, redundancy_sigma)
            best_individuals = population[places][-population_survival_size:]
            best_cross = cross_section(best_individuals)
            population = np.squeeze(mutate.from_adjacency_matrix(
                adjacency_matrix=best_cross,
                prob=reproduction_mutation_prob,
                n=1,
            ))
            scores = np.sort(scores)
            if np.isclose(previous_best, scores[-1], atol=10e-5):
                n_repeats += 1
            else:
                n_repeats = 0
            previous_best = scores[-1]
            if n_repeats > 3:
                reproduction_mutation_prob *= 2
            else:
                reproduction_mutation_prob = initial_reproduction_mutation_prob

            print(f"{generation=}::{n_repeats=}::{np.sum(population[places][-1])}::{reproduction_mutation_prob=}::{scores[-5:]}")
        _, places = self.evaluate(population, redundancy_level, redundancy_sigma)
        self.clear_edges()
        self.update(nx.from_numpy_array(population[places][-1]))
        return population[places][-1]
