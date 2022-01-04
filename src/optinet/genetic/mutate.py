from typing import Union, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt


def check_for_adjacency_matrix(adjacency_matrix: npt.NDArray[np.int_], ) -> bool:
    if not isinstance(adjacency_matrix, np.ndarray):
        raise TypeError(f"An adjacency_matrix of type numpy.ndarray is expected, "
                        f"{type(adjacency_matrix)} is obtained instead.")
    if adjacency_matrix.ndim < 2:
        raise ValueError("Given adjacency_matrix has less then 2 dimensions.")
    if not adjacency_matrix.shape[-1] == adjacency_matrix.shape[-2]:
        raise ValueError("Given matrix or matrices are not square.")
    if np.any(adjacency_matrix > 1):
        raise NotImplementedError("Hypergraph mutation is not yet supported.")
    return True


def from_adjacency_matrix(adjacency_matrix: npt.NDArray[np.int_],
                          prob: Union[float, int] = 0.1,
                          rng: Optional[np.random.Generator] = None,
                          n: int = 1) -> npt.NDArray[np.int_]:
    """Generates mutations in the adjacency matrix with the given mutation probability.

    Parameters
    ----------
    adjacency_matrix
        The adjacency matrix to be mutated.
    prob
        Edge mutation probability. Default is 0.1. Must satisfy the following inequality:
        float: 0.0 <= prob <= 1.0
        int: 0 <= prob <= 100
        If an integer value is given, it is interpreted as a percentage.
    rng
        Numpy random generator. Default is np.random.default_rng.
    n
        Number of mutations to generate.

    Returns
    -------
    numpy.ndarray
        n mutated adjacency matrices of the same shape as an input matrix.
    """
    if isinstance(prob, int):
        prob /= 100
    if prob > 1:
        raise ValueError("Given probability is greater than 1 (or 100%)")
    if rng is None:
        rng = np.random.default_rng()

    if check_for_adjacency_matrix(adjacency_matrix):
        masks = rng.choice([0, 1], size=(n, *adjacency_matrix.shape), p=(1 - prob, prob))
        masks = np.tril(masks, k=-1)
        masks += np.transpose(masks, axes=(0, 2, 1))
        mut_adjacency_matrix: npt.NDArray[np.int_] = adjacency_matrix.astype(int) ^ masks
        return mut_adjacency_matrix


def from_graph(graph: nx.Graph,
               prob: Union[float, int] = 0.1,
               rng: Optional[np.random.Generator] = None) -> nx.Graph:
    """Generates mutated graph with the given mutation probability.

    Parameters
    ----------
    graph
        The graph to be mutated.
    prob
        Edge mutation probability. Default is 0.1. Must satisfy the following inequality:
        float: 0.0 <= prob <= 1.0
        int: 0 <= prob <= 100
        If an integer value is given, it is interpreted as a percentage.
    rng
        Numpy random generator. Default is np.random.default_rng.

    Returns
    -------
    networkx.Graph
        Mutated graph.

    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"A graph of type networkx.classes.graph.Graph is expected, "
                        f"{type(graph)} is obtained instead.")
    elif isinstance(graph, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("Types DiGraph, MultiGraph and MultiDiGraph are not implemented yet.")

    adjacency_matrix = nx.to_numpy_array(graph, dtype=int)
    return nx.from_numpy_array(from_adjacency_matrix(adjacency_matrix, prob, rng)[0])
