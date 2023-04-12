"""
Simple Example
"""
import numpy as np
from scipy.stats import distributions as ds  # type: ignore
from scipy.stats import norm, uniform # type: ignore


def monomial_1D(
    p: int = 5,
    n_samples: int = int(1e3),
    mean: float = 0.25,
    std_dev: float = 0.1,
    N: int = 1,
):
    """
    1D Monomial Map over [-1, 1].

    Parameters
    ----------
    p: int, default=5
        Power of monomial.
    n_samples : int, default=1000
        Number of samples to draw from uniform initial over domain.
    mu: float, default=0.25
        True mean value of observed data.
    sigma: float, default=0.1
        Standard deviation of observed data.
    N: int, default=1
        Number of data points to generate from observed distribution. Note if 1,
        the default value, then the singular drawn value will always be ``mu``.

    Returns
    -------
    data: Tuple[:class:`numpy.ndarray`,]
        Tuple of ``(lam, q_lam, data)`` where ``lam`` is contains the
        :math:`\lambda` samples, ``q_lam`` the value of :math:`Q_p(\lambda)`,
        and ``data`` the observed data values from the
        :math:`\mathcal{N}(\mu, \sigma)` distribution.

    Examples
    --------
    Note when N=1, data point drawn is always equal to mean.

    >>> import numpy as np
    >>> from pydcia.examples.monomial import monomial_1D
    >>> lam, q_lam, data, std_dev = polynomial_1D_data(num_samples=10, N=1)
    >>> data[0]
    0.25
    >>> len(lam)
    10

    For higher values of N, values are drawn from N(mean, std_dev) distribution.

    >>> lam, q_lam, data = polynomial_1D_data(N=10, mean=0.5, std_dev=0.01)
    >>> len(data)
    10
    >>> np.mean(data) < 0.6
    True
    """
    if N == 1:
        data = np.array([mean])
    else:
        data = norm.rvs(loc=mean, scale=std_dev**2, size=N)
    lam = uniform.rvs(size=(n_samples, 1), loc=-1, scale=2)
    q_lam = (lam**p).reshape(n_samples, -1)
    pi_obs = norm(data, scale=std_dev)
    return lam, q_lam, data, std_dev

