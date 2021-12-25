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

    def set_edge_bandwidths(self):
        """Set bandwidth for each edge."""
        if not self.edges:
            raise RuntimeWarning("The graph has no edges, no edge bandwidths were set.")
        else:
            bandwidths = self.multi_incidence_matrix.sum(axis="columns")
            nx.set_edge_attributes(self, dict(zip(self.edges, bandwidths)), name="bandwidth")

    @property
    def total_length(self) -> float:
        """Total length of all edges."""
        if self.edges:
            self.set_edge_lengths()
        self._total_length = sum(nx.get_edge_attributes(self, "length").values())
        return self._total_length

    # TODO: this property is not more necessary, remove it
    @property
    def multi_incidence_matrix(self):
        """A matrix with complete description of how much traffic each edge carries to each node."""
        current_index = pd.MultiIndex.from_tuples(list(self.edges), names=["n1", "n2"])
        if self._multi_incidence_matrix is None:
            if self.edges:
                data = np.ones(shape=(len(current_index), len(self.nodes)))
                df = pd.DataFrame(data, columns=np.array(self.nodes).astype(str), index=current_index)
                self._multi_incidence_matrix = df
        else:
            old_index = self._multi_incidence_matrix.index
            if len(old_index) != len(current_index) or (old_index != current_index).any():
                new_matrix = pd.DataFrame(index=current_index).join(
                    self._multi_incidence_matrix,
                    how="inner",
                )
                index_diff = set(current_index).difference(old_index)
                data = np.ones(shape=(len(index_diff), len(self.nodes)))
                df = pd.DataFrame(data, columns=np.array(self.nodes).astype(str), index=index_diff)
                self._multi_incidence_matrix = new_matrix.append(df).sort_index()
        return self._multi_incidence_matrix

    @multi_incidence_matrix.setter
    def multi_incidence_matrix(self, value: Optional[pd.DataFrame]):
        if value is None and self.edges:
            raise ValueError("multi_incidence_matrix can not be None, when graph contains edges")
        elif value is None:     # and not self.edges
            self._multi_incidence_matrix = None
        elif self.edges and value.shape == (len(self.edges), len(self.nodes)):
            acceptable_index = pd.MultiIndex.from_tuples(list(self.edges), names=["n1", "n2"])
            if (value.sort_index().index == acceptable_index).all():
                self._multi_incidence_matrix = value
                self.set_edge_bandwidths()
            else:
                raise ValueError("Index of the DataFrame is not compatible with the current graph state.")
        else:
            raise ValueError(f"Shape of the multi_incidence_matrix must be equal (# of edges, # of nodes). "
                             f"For this graph: ({len(self.edges)}, {len(self.nodes)}). "
                             f"{value.shape} obtained instead.")

    def optimise(self, n_mutations=100, prob=0.8, **kwargs):
        for i in range(1, n_mutations+1):
            new_graph = self.copy()
            new_graph.clear_edges()
            new_graph.update(mutate.from_graph(self, prob=(prob/i), **kwargs))
            if new_graph.total_length < self.total_length and nx.is_connected(new_graph):
                self.clear_edges()
                self.update(new_graph)
