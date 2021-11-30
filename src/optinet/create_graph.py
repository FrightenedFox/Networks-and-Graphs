from typing import Union
import networkx as nx
import pandas as pd
from geopy.distance import distance


class CitiesNodes(nx.Graph):

    def __init__(self, node_attrs: Union[dict, pd.DataFrame] = None, **kwargs):
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

        if isinstance(node_attrs, pd.DataFrame):
            self.add_nodes_from(node_attrs.index)
            nx.set_node_attributes(self, node_attrs.to_dict("index"))
        elif isinstance(node_attrs, dict):
            self.add_nodes_from(node_attrs.keys())
            nx.set_node_attributes(self, node_attrs)
        elif node_attrs is not None:
            raise TypeError(f"node_attrs of type dictionary or pandas.DataFrame is expected, "
                            f"{type(node_attrs)} is obtained instead.")

    def calculate_edge_length(self, edge):
        # TODO: replace with more elegant code
        longitudes = nx.get_node_attributes(self, "Longitude")
        latitude = nx.get_node_attributes(self, "Latitude")
        a, b = edge
        coord_a = [latitude[a], longitudes[a]]
        coord_b = [latitude[b], longitudes[b]]
        return distance(coord_a, coord_b).km

    def set_edge_lengths(self):
        pass

    # TODO: rewrite original method for adding edges
    #       super().add_edges_from() should help
