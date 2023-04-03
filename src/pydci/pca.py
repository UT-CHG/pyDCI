"""
MUD Pre-Processing Module

All functions for pre-processing QoI data-sets before applying inversion
algorithms can be found in this module.

Functions
---------
pca - Applly Principle Component Analysis transformation to QoI data.

"""

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def pca(data: ArrayLike, n_components: int = 2, **kwargs) -> Tuple[PCA, np.ndarray]:
    """
    Apply Principal Component Analysis

    Uses :class:`sklearn.decomposition.PCA` class to perform a truncated PCA
    transformation on input ``data`` using the first ``n_components`` principle
    components. Note :class:`sklearn.preprocessing.StandardScaler`
    transformation is applied to the data first.


    Parameters
    ----------
    ds : :obj:`numpy.typing.ArrayLike`
        Data to apply PCA transformation to. Must be 2 dimensional.
    n_components: int, default=2
        Number of principal components to use.
    kwargs: dict, optional
        Additional keyword arguments will be passed to
        :class:`sklearn.decomposition.PCA` class constructor. See sklearn's
        documentation for more information on how PCA is performed.

    Returns
    -------
    pca_res: Tuple[:class:`sklearn.decomposition.PCA`, :class:`numpy.ndarray`]
        Tuple of ``(pca, X_train)`` where ``pca`` is the
        :class:`sklearn.decomposition.PCA` class with principle component
        vectors accessible at ``pca.components_`` and ``X_train`` being the
        transformed data-set, which should have same number of rows as original
        ``data``, but now only ``n_components`` columns.

    Examples
    --------

    For a simple example lets apply the PCA transformation to the identity
    matrix in 2 dimensions, using first 1 principle component.

    >>> data = np.eye(2)
    >>> pca_1, X_train_1 = pca(data, n_components=1)
    >>> np.around(X_train_1, decimals=1)
    array([[-1.4],
           [ 1.4]])
    >>> np.around(pca_1.components_, decimals=1)
    array([[-0.7,  0.7]])

    Now lets try using two components

    >>> pca_2, X_train_2 = pca(data, n_components=2)
    >>> np.around(X_train_2, decimals=1)
    array([[-1.4,  0. ],
           [ 1.4,  0. ]])
    >>> np.abs(np.around(pca_2.components_, decimals=1))
    array([[0.7, 0.7],
           [0.7, 0.7]])

    Note that if we have three dimensional data we must flatten it before
    sending using ``pca()``

    >>> data = np.random.rand(2,2,2)
    >>> pca, X_train = pca(data)
    Traceback (most recent call last):
        ...
    ValueError: Data is 3 dimensional. Must be 2D

    Assuming the first dimension indicates each sample, and each sample contains
    2D data within the 2nd and 3rd dimensions of the of the data set, then we
    can flatten this 2D data into a vector and then perform the PCA
    transformation.

    >>> data = np.reshape(data, (2,-1))
    >>> pca, X_train = pca(data)
    >>> X_train.shape
    (2, 2)

    """
    ndim = np.array(data).ndim
    if ndim != 2:
        raise ValueError(f"Data is {ndim} dimensional. Must be 2D")

    # Standarize and perform linear PCA
    sc = StandardScaler()
    pca = PCA(n_components=n_components, **kwargs)
    X_train = pca.fit_transform(sc.fit_transform(data))

    return pca, X_train


def svd(data: ArrayLike, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Singular Value Decomposition

    Uses :class:`np.linalg.svd` class to perform an SVD transformation on input
    ``data``. Note :class:`sklearn.preprocessing.StandardScaler`
    transformation is applied to the data first.


    Parameters
    ----------
    ds : :obj:`numpy.typing.ArrayLike`
        Data to apply SVD transformation to. Must be 2 dimensional.
    kwargs: dict, optional
        Additional keyword arguments will be passed to
        :class:`np.linalg.svd` method.

    Returns
    -------
    svd_res: Tuple[:class:`numpy.ndarray`,]
        Tuple of ``(U, singular_values, singular_vectors)`` corresponding to
        the X = Sigma UV^T decomposition elements.

    Examples
    --------

    For a simple example lets apply the PCA transformation to the identity
    matrix in 2 dimensions, using first 1 principle component.

    >>> data = np.eye(2)
    >>> U, S, V = svd(data)
    >>> np.around(U, decimals=1)
    array([[-0.7,  0.7],
           [ 0.7,  0.7]])
    >>> np.around(S, decimals=1)
    array([2., 0.])

    Note that if we have three dimensional data we must flatten it before
    sending using ``pca()``

    >>> data = np.random.rand(2,2,2)
    >>> U, S, V = svd(data)
    Traceback (most recent call last):
        ...
    ValueError: Data is 3 dimensional. Must be 2D

    Assuming the first dimension indicates each sample, and each sample contains
    2D data within the 2nd and 3rd dimensions of the of the data set, then we
    can flatten this 2D data into a vector and then perform ``svd()``.

    >>> data = np.reshape(data, (2,-1))
    >>> U, S, V = svd(data)
    >>> U.shape
    (2, 2)

    """
    ndim = np.array(data).ndim
    if ndim != 2:
        raise ValueError(f"Data is {ndim} dimensional. Must be 2D")

    # Standarize and perform SVD
    sc = StandardScaler()
    X = sc.fit_transform(data)
    U, S, V = np.linalg.svd(X)

    return (U, S, V)
