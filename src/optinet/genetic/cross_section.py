from typing import Optional

import numpy as np
import numpy.typing as npt


def cross_section(adjacency_matrices: npt.NDArray[np.int_],
                  rng: Optional[np.random.Generator] = None):
    """

    Parameters
    ----------
    adjacency_matrices :
    rng :

    Returns
    -------

    """
    if rng is None:
        rng = np.random.default_rng()
    masks = rng.choice([0, 1], size=adjacency_matrices.shape)
    masks = np.tril(masks, k=-1)
    masks += np.transpose(masks, axes=(0, 2, 1))
    x = adjacency_matrices[:, None, ...] & masks
    y = adjacency_matrices & ~ masks
    return (x | y).reshape(adjacency_matrices.shape[0] ** 2, *adjacency_matrices.shape[1:3])
