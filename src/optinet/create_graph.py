from typing import Union, Dict, Any

import numpy as np
import networkx as nx
import pandas as pd
from geopy.distance import distance


class CitiesNodes(nx.Graph):

    def __init__(self, node_attrs: Union[Dict[int, Dict[str, Any]], pd.DataFrame] = None, **kwargs):
        """
        Creates networkx Graph with node attributes if passed.

        Parameters
        ----------
        node_attrs : (optional)
            Dictionary or Pandas DataFrame with information about the nodes and attributes to be added to the graph.
        kwargs :
            networkx.Graph arguments.
        """
        super().__init__(**kwargs)

        self._total_length = None

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
        """ Calculate length for each edge in edges.

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

    def set_edge_lengths(self) -> None:
        """ Set the length for each edge. """
        if not self.edges:
            raise RuntimeError("Graph has no edges")
        lengths = self.calculate_edge_lengths(list(self.edges))
        nx.set_edge_attributes(self, dict(zip(self.edges, lengths)), name="length")

    # NOTE: think of rewriting original method for adding edges
    #       super().add_edges_from() should help

    @property
    def total_length(self):
        """ Total length of all edges. """
        self._total_length = sum(nx.get_edge_attributes(self, "length").values())
        return self._total_length
