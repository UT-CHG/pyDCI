"""
PyDCI Utils

"""
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


def add_noise(signal: ArrayLike, sd: float = 0.05, seed: int = None):
    """
    Add Noise

    Add noise to synthetic signal to model a real measurement device. Noise is
    assumed to be from a standard normal distribution std deviation `sd`:

    $\\mathcal{N}(0,\\sigma)$

    Paramaters
    ---------
    signal : numpy.typing.ArrayLike
      Signal to add noise to.
    sd : float, default = 0.05
      Standard deviation of error to add.
    seed : int, optional
      Seed to use for numpy random number generator.

    Returns
    -------
    noisy_signal: numpy.typing.ArrayLike
      Signal with noise added to it.

    Example Usage
    -------------
    Generate test signal, add noise, check average distance
    >>> seed = 21
    >>> test_signal = np.ones(5)
    >>> noisy_signal = add_noise(test_signal, sd=0.05, seed=21)
    >>> np.round(1000*np.mean(noisy_signal-test_signal))
    4.0
    """
    signal = np.array(signal)

    if seed is not None:
        np.random.seed(seed)

    # Populate qoi_true with noise
    noise = np.random.randn(signal.size) * sd

    return signal + noise


def rank_decomposition(A: np.typing.ArrayLike) -> List[np.ndarray]:
    """Build list of rank k updates of A"""
    A = np.array(A)
    A_ranks = []
    rank_1_updates = []
    u, s, v = np.linalg.svd(A)
    A_ranks.append(s[0] * (u[:, 0].reshape(-1, 1)) @ v[:, 0].reshape(1, -1))
    for i in range(1, A.shape[1]):
        rank_1_updates.append(s[i] * (u[:, i].reshape(-1, 1)) @ v[:, i].reshape(1, -1))
        A_ranks.append(sum(rank_1_updates[0:i]))

    return A_ranks


def fit_domain(
    x: np.ndarray = None, min_max_bounds: np.ndarray = None, pad_ratio: float = 0.1
) -> np.ndarray:
    """
    Fit domain bounding box to array x

    Parameters
    ----------
    x : ArrayLike
        2D array to calculate min, max values along columns.
    pad_ratio : float, default=0.1
        What ratio of total range=max-min to subtract/add to min/max values to
        construct final domain range. Padding is done per x column dimension.

    Returns
    -------
    min_max_bounds : ArrayLike
        Domain fitted to values found in 2D array x, with padding added.

    Examples
    --------
    Input must be 2D. Set pad_ratio = 0 to get explicit min/max bounds
    >>> fit_domain(np.array([[1, 10], [0, -10]]), pad_ratio=0.0)
    array([[  0,   1],
           [-10,  10]])

    Can extend domain around the array values using the pad_ratio argument.

    >>> fit_domain(np.array([[1, 10], [0, -10]]), pad_ratio=1)
    array([[ -1,   2],
           [-30,  30]])
    """
    if min_max_bounds is None:
        if x is None:
            raise ValueError("Both x and min_max_bounds can't be None")
        min_max_bounds = np.array([x.min(axis=0), x.max(axis=0)]).T
    pad = pad_ratio * (min_max_bounds[:, 1] - min_max_bounds[:, 0])
    min_max_bounds[:, 0] = min_max_bounds[:, 0] - pad
    min_max_bounds[:, 1] = min_max_bounds[:, 1] + pad
    return min_max_bounds


def set_shape(array: np.ndarray, shape: Union[List, Tuple] = (1, -1)) -> np.ndarray:
    """Resizes inputs if they are one-dimensional."""
    return array.reshape(shape) if array.ndim < 2 else array


def get_uniform_box(self, center, factor=0.5, mins=None, maxs=None):
    """
    Generate a domain of [min, max] values around a center value.
    """
    center = np.array(center)
    if np.sum(center) != 0.0:
        loc = center - np.abs(center * factor)
        scale = 2 * center * factor
    else:
        loc = center - factor
        scale = 2 * factor
    domain = np.array(list(zip(loc, np.array(loc) + np.array(scale))))
    if mins is not None:
        for i, d in enumerate(domain):
            if d[0] < mins[i]:
                d[0] = mins[i]
    if maxs is not None:
        for i, d in enumerate(domain):
            if d[1] > maxs[i]:
                d[1] = maxs[i]

    return domain

