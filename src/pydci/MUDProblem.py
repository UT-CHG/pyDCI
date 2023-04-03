from typing import Callable, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import distributions as dist  # type: ignore
from scipy.stats import gaussian_kde as gkde  # type: ignore
from scipy.stats import rv_continuous  # type: ignore

from pydci.utils import fit_domain, set_shape


class MUDProblem(object):
    """
    Sets up Data-Consistent Inverse Problem for parameter identification

    Data-Consistent inversion is a way to infer most likely model parameters
    using observed data and predicted data from the model.

    Attributes
    ----------
    X : ArrayLike
        Array containing parameter samples from an initial distribution.
        Rows represent each sample while columns represent parameter values.
        If 1 dimensional input is passed, assumed that it represents repeated
        samples of a 1-dimensional parameter.
    y : ArrayLike
        Array containing push-forward values of parameters samples through the
        forward model. These samples will form the `predicted distribution`.
    domain : ArrayLike
        Array containing ranges of each parameter value in the parameter
        space. Note that the number of rows must equal the number of
        parameters, and the number of columns must always be two, for min/max
        range.
    weights : ArrayLike, optional
        Weights to apply to each parameter sample. Either a 1D array of the
        same length as number of samples or a 2D array if more than
        one set of weights is to be incorporated. If so the weights will be
        multiplied and normalized row-wise, so the number of columns must
        match the number of samples.

    Examples
    -------------

    Generate test 1-D parameter estimation problem. Model to produce predicted
    data is the identity map and observed signal comes from true value plus
    some random gaussian nose.

    See :meth:`mud.examples.identity_uniform_1D_density_prob` for more details

    >>> from pydci.examples.simple import identity_1D_density_prob as I1D

    First we set up a well-posed problem. Note the domain we are looking over
    contains our true value. We take 1000 samples, use 50 observations,
    assuming a true value of 0.5 populated with gaussian noise
    :math:`\\mathcal{N}(0,0.5)`. Or initial uniform distribution is taken from a
    :math:`[0,1]` range.

    >>> D = I1D(1000, 50, 0.5, 0.05, domain=[0,1])

    Estimate mud_point -> Note since WME map used, observed implied to be the
    standard normal distribution and does not have to be set explicitly from
    observed data set.

    >>> np.round(D.mud_point()[0],1)
    0.5

    Expectation value of r, ratio of observed and predicted distribution, should
    be near 1 if predictability assumption is satisfied.

    >>> np.round(D.expected_ratio(),0)
    1.0

    Set up ill-posed problem -> Searching out of range of true value

    >>> D = I1D(1000, 50, 0.5, 0.05, domain=[0.6,1])

    Mud point will be close as we can get within the range we are searching for

    >>> np.round(D.mud_point()[0],1)
    0.6

    Expectation of r is close to zero since predictability assumption violated.

    >>> np.round(D.expected_ratio(),1)
    0.0

    """

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        domain: ArrayLike = None,
        weights: ArrayLike = None,
        normalize: bool = False,
        pad_domain: float = 0.1,
    ):

        self.X = set_shape(np.array(X), (1, -1))
        self.y = set_shape(np.array(y), (-1, 1))

        # These will be updated in set_ and fit() functions
        self._r = None  # Ratio of observed to predicted
        self._up = None  # Updated values
        self._in = None  # Initial values
        self._pr = None  # Predicted values
        self._ob = None  # Observed values
        self._in_dist = None  # Initial distribution
        self._pr_dist = None  # Predicted distribution
        self._ob_dist = None  # Observed distribution

        if domain is not None:
            # Assert domain passed in is consistent with data array
            self.domain = set_shape(np.array(domain), (1, -1))
            assert (
                self.domain.shape[0] == self.n_params
            ), f"Size mismatch: domain: {self.domain.shape}, params: {self.X.shape}"

        else:
            self.domain = fit_domain(self.X)

        # Initialize weights
        self.set_weights(weights, normalize=normalize)

    @property
    def n_params(self):
        return self.X.shape[1]

    @property
    def n_features(self):
        return self.y.shape[1]

    @property
    def n_samples(self):
        return self.y.shape[0]

    def set_weights(self, weights: ArrayLike = None, normalize: bool = False):
        """Set Sample Weights

        Sets the weights to use for each sample. Note weights can be one or two
        dimensional. If weights are two dimensional the weights are combined
        by multiplying them row wise and normalizing, to give one weight per
        sample. This combining of weights allows incorporating multiple sets
        of weights from different sources of prior belief.

        Parameters
        ----------
        weights : np.ndarray, List
            Numpy array or list of same length as the `n_samples` or if two
            dimensional, number of columns should match `n_samples`
        normalize : bool, default=False
            Whether to normalize the weights vector.

        Returns
        -------

        Warnings
        --------
        Resetting weights will delete the predicted and updated distribution
        values in the class, requiring a re-run of adequate `set_` methods
        and/or `fit()` to reproduce with new weights.
        """
        if weights is None:
            w = np.ones(self.X.shape[0])
        else:
            w = np.array(weights)

            # Reshape to 2D
            w = w.reshape(1, -1) if w.ndim == 1 else w

            # assert appropriate size
            assert self.n_samples == w.shape[1], f"`weights` must size {self.n_samples}"

            # Multiply weights column wise for stacked weights
            w = np.prod(w, axis=0)

            if normalize:
                w = np.divide(w, np.linalg.norm(w))

        self._weights = w
        self._pr = None
        self._up = None
        self._pr_dist = None

    def set_observed(self, distribution: rv_continuous = dist.norm()):
        """Set distribution for the observed data.

        The observed distribution is determined from assumptions on the
        collected data. In the case of using a weighted mean error map on
        sequential data from a single output, the distribution is stationary
        with respect to the number data points collected and will always be
        the standard normal d distribution $N(0,1)$.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, default=scipy.stats.norm()
            scipy.stats continuous distribution like object representing the
            likelihood of observed data. Defaults to a standard normal
            distribution N(0,1).

        """
        self._ob_dist = distribution
        self._ob = distribution.pdf(self.y).prod(axis=1)

    def set_initial(self, distribution: Optional[rv_continuous] = None):
        """
        Set initial probability distribution of model parameter values
        :math:`\\pi_{in}(\\lambda)`.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, optional
            scipy.stats continuous distribution object from where initial
            parameter samples were drawn from. If none provided, then a uniform
            distribution over domain of the density problem is assumed. If no
            domain is specified for density, then a standard normal
            distribution :math:`N(0,1)` is assumed.

        Warnings
        --------
        Setting initial distribution resets the predicted and updated
        distributions, so make sure to set the initial first.
        """
        if distribution is None:  # assume standard normal by default
            mn = np.min(self.domain, axis=1)
            mx = np.max(self.domain, axis=1)
            distribution = dist.uniform(loc=mn, scale=mx - mn)

        self._in = distribution.pdf(self.X).prod(axis=1)
        self._in_dist = distribution
        self._up = None
        self._pr = None
        self._pr_dist = None

    def set_predicted(
        self,
        distribution: rv_continuous = None,
        bw_method: Union[str, Callable, np.generic] = None,
        weights: ArrayLike = None,
        **kwargs,
    ):
        """
        Set Predicted Distribution

        The predicted distribution over the observable space is equal to the
        push-forward of the initial through the model
        :math:`\\pi_{pr}(Q(\\lambda)`. If no distribution is passed,
        :class:`scipy.stats.gaussian_kde` is used over the predicted values
        :attr:`y` to estimate the predicted distribution.

        Parameters
        ----------
        distribution : :class:`scipy.stats.rv_continuous`, optional
            If specified, used as the predicted distribution instead of the
            default of using gaussian kernel density estimation on observed
            values y. This should be a frozen distribution if using
            `scipy`, and otherwise be a class containing a `pdf()` method
            return the probability density value for an array of values.
        bw_method : str, scalar, or `Callable`, optional
            Method to use to calculate estimator bandwidth. Only used if
            distribution is not specified, See documentation for
            :class:`scipy.stats.gaussian_kde` for more information.
        weights : ArrayLike, optional
            Weights to use on predicted samples. Note that if specified,
            :meth:`set_weights` will be run first to calculate new weights.
            Otherwise, whatever was previously set as the weights is used.
            Note this defaults to a weights vector of all 1s for every sample
            in the case that no weights were passed on upon initialization.
        **kwargs: dict, optional
            If specified, any extra keyword arguments will be passed along to
            the passed ``distribution.pdf()`` function for computing values of
            predicted samples.

        Note: `distribution` should be a frozen distribution if using `scipy`.

        Warnings
        --------
        If passing a `distribution` argument, make sure that the initial
        distribution has been set first, either by having run
        :meth:`set_initial` or :meth:`fit` first.
        """
        if weights is not None:
            self.set_weights(weights)

        if distribution is None:
            # Reweight kde of predicted by weights if present
            distribution = gkde(self.y.T, bw_method=bw_method,
                                weights=self._weights)
            pred_pdf_values = distribution.pdf(self.y.T).T
        else:
            pred_pdf_values = distribution.pdf(self.y, **kwargs)

        self._pr_dist = distribution
        self._pr = pred_pdf_values.ravel()
        self._up = None

    def fit(self, **kwargs):
        """
        Update Initial Distribution

        Constructs the updated distribution by fitting observed data to
        predicted data with:

        .. math::
            \\pi_{up}(\\lambda) = \\pi_{in}(\\lambda)
            \\frac{\\pi_{ob}(Q(\\lambda))}{\\pi_{pred}(Q(\\lambda))}
            :label: data_consistent_solution

        Note that if initial, predicted, and observed distributions have not
        been set before running this method, they will be run with default
        values. To set specific predicted, observed, or initial distributions
        use the ``set_`` methods.

        Parameters
        -----------
        **kwargs : dict, optional
            If specified, optional arguments are passed to the
            :meth:`set_predicted` call in the case that the predicted
            distribution has not been set yet.

        Returns
        -----------

        """
        if self._in is None:
            self.set_initial()
        if self._pr is None:
            self.set_predicted(**kwargs)
        if self._ob is None:
            self.set_observed()

        # Store ratio of observed/predicted
        # e.g. to comptue E(r) and to pass on to future iterations
        self._r = np.divide(self._ob, self._pr)

        # Multiply by initial to get updated pdf
        up_pdf = np.multiply(self._in * self._weights, self._r)
        self._up = up_pdf

    def mud_point(self):
        """Maximal Updated Density (MUD) Point

        Returns the Maximal Updated Density or MUD point as the parameter
        sample from the initial distribution with the highest update density
        value:

        .. math::
            \\lambda^{MUD} := \\mathrm{argmax} \\pi_{up}(\\lambda)
            :label: mud

        Note if the updated distribution has not been computed yet, this
        function will call :meth:`fit` to compute it.

        Parameters
        ----------

        Returns
        -------
        mud_point : np.ndarray
            Maximal Updated Density (MUD) point.
        """
        if self._up is None:
            self.fit()
        m = np.argmax(self._up)
        return self.X[m, :]

    def estimate(self):
        """Estimate

        Returns the best estimate for most likely parameter values for the
        given model data using the data-consistent framework.

        Parameters
        ----------

        Returns
        -------
        mud_point : np.ndarray
            Maximal Updated Density (MUD) point.
        """
        return self.mud_point()

    def expected_ratio(self):
        """Expectation Value of R

        Returns the expectation value of the R, the ratio of the observed to
        the predicted density values.

        .. math::
            R = \\frac{\\pi_{ob}(\\lambda)}
                      {\\pi_{pred}(\\lambda)}
            :label: r_ratio

        If the predictability assumption for the data-consistent framework is
        satisfied, then :math:`E[R]\\approx 1`.

        Parameters
        ----------

        Returns
        -------
        expected_ratio : float
            Value of the E(r). Should be close to 1.0.
        """
        if self._up is None:
            self.fit()

        return np.average(self._r, weights=self._weights)


class MUDProblem2(object):
    """
    Sets up Data-Consistent Inverse Problem for parameter identification

    Data-Consistent inversion is a way to infer most likely model parameters
    using observed data and predicted data from the model.

    Attributes
    ----------
    samples : ArrayLike
        Array containing parameter samples from an initial distribution.
        Rows represent each sample while columns represent parameter values.
        If 1 dimensional input is passed, assumed that it represents repeated
        samples of a 1-dimensional parameter.
    model : Model
        Model to push forward samples through
    domain : ArrayLike
        Array containing ranges of each parameter value in the parameter
        space. Note that the number of rows must equal the number of
        parameters, and the number of columns must always be two, for min/max
        range. If non specified, will be inferred from the sampls array.
    weights : ArrayLike, optional
        Weights to apply to each parameter sample. Either a 1D array of the
        same length as number of samples or a 2D array if more than
        one set of weights is to be incorporated. If so the weights will be
        multiplied and normalized row-wise, so the number of columns must
        match the number of samples.
    """

    def __init__(
        self,
        model,
        samples,
        domain: ArrayLike = None,
        weights: ArrayLike = None,
        pad_domain: float = 0.1,
        normalize: bool = False,
    ):

      #         self.samples = set_shape(np.array(X), (1, -1))
      #         self.y = set_shape(np.array(y), (-1, 1))
      # 
      #         # These will be updated in set_ and fit() functions
      #         self._r = None  # Ratio of observed to predicted
      #         self._up = None  # Updated values
      #         self._in = None  # Initial values
      #         self._pr = None  # Predicted values
      #         self._ob = None  # Observed values
      #         self._in_dist = None  # Initial distribution
      #         self._pr_dist = None  # Predicted distribution
      #         self._ob_dist = None  # Observed distribution

        self.samples = samples
        if domain is not None:
            # Assert domain passed in is consistent with data array
            self.domain = set_shape(np.array(domain), (1, -1))
            assert (
                self.domain.shape[0] == self.n_params
            ), f"Size mismatch: domain: {self.domain.shape}, params: {self.X.shape}"

        else:
            self.domain = fit_domain(self.X)

        # Initialize weights
        self.set_weights(weights, normalize=normalize)

    @property
    def n_params(self):
        return self.model.n_params

    @property
    def n_states(self):
        return self.model.n_states

    @property
    def n_samples(self):
        return self.samples.shape[0]

    def set_weights(self, weights: ArrayLike = None, normalize: bool = False):
        """Set Sample Weights

        Sets the weights to use for each sample. Note weights can be one or two
        dimensional. If weights are two dimensional the weights are combined
        by multiplying them row wise and normalizing, to give one weight per
        sample. This combining of weights allows incorporating multiple sets
        of weights from different sources of prior belief.

        Parameters
        ----------
        weights : np.ndarray, List
            Numpy array or list of same length as the `n_samples` or if two
            dimensional, number of columns should match `n_samples`
        normalize : bool, default=False
            Whether to normalize the weights vector.

        Returns
        -------

        Warnings
        --------
        Resetting weights will delete the predicted and updated distribution
        values in the class, requiring a re-run of adequate `set_` methods
        and/or `fit()` to reproduce with new weights.
        """
        if weights is None:
            w = np.ones(self.X.shape[0])
        else:
            w = np.array(weights)

            # Reshape to 2D
            w = w.reshape(1, -1) if w.ndim == 1 else w

            # assert appropriate size
            assert self.n_samples == w.shape[1], f"`weights` must size {self.n_samples}"

            # Multiply weights column wise for stacked weights
            w = np.prod(w, axis=0)

            if normalize:
                w = np.divide(w, np.linalg.norm(w))

        self._weights = w
        self._pr = None
        self._up = None
        self._pr_dist = None

    def set_observed(self, distribution: rv_continuous = dist.norm()):
        """Set distribution for the observed data.

        The observed distribution is determined from assumptions on the
        collected data. In the case of using a weighted mean error map on
        sequential data from a single output, the distribution is stationary
        with respect to the number data points collected and will always be
        the standard normal d distribution $N(0,1)$.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, default=scipy.stats.norm()
            scipy.stats continuous distribution like object representing the
            likelihood of observed data. Defaults to a standard normal
            distribution N(0,1).

        """
        self._ob_dist = distribution
        self._ob = distribution.pdf(self.y).prod(axis=1)

    def set_initial(self, distribution: Optional[rv_continuous] = None):
        """
        Set initial probability distribution of model parameter values
        :math:`\\pi_{in}(\\lambda)`.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, optional
            scipy.stats continuous distribution object from where initial
            parameter samples were drawn from. If none provided, then a uniform
            distribution over domain of the density problem is assumed. If no
            domain is specified for density, then a standard normal
            distribution :math:`N(0,1)` is assumed.

        Warnings
        --------
        Setting initial distribution resets the predicted and updated
        distributions, so make sure to set the initial first.
        """
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()

        self._in = distribution.pdf(self.X).prod(axis=1)
        self._in_dist = distribution
        self._up = None
        self._pr = None
        self._pr_dist = None

    def set_predicted(
        self,
        distribution: rv_continuous = None,
        bw_method: Union[str, Callable, np.generic] = None,
        weights: ArrayLike = None,
        **kwargs,
    ):
        """
        Set Predicted Distribution

        The predicted distribution over the observable space is equal to the
        push-forward of the initial through the model
        :math:`\\pi_{pr}(Q(\\lambda)`. If no distribution is passed,
        :class:`scipy.stats.gaussian_kde` is used over the predicted values
        :attr:`y` to estimate the predicted distribution.

        Parameters
        ----------
        distribution : :class:`scipy.stats.rv_continuous`, optional
            If specified, used as the predicted distribution instead of the
            default of using gaussian kernel density estimation on observed
            values y. This should be a frozen distribution if using
            `scipy`, and otherwise be a class containing a `pdf()` method
            return the probability density value for an array of values.
        bw_method : str, scalar, or `Callable`, optional
            Method to use to calculate estimator bandwidth. Only used if
            distribution is not specified, See documentation for
            :class:`scipy.stats.gaussian_kde` for more information.
        weights : ArrayLike, optional
            Weights to use on predicted samples. Note that if specified,
            :meth:`set_weights` will be run first to calculate new weights.
            Otherwise, whatever was previously set as the weights is used.
            Note this defaults to a weights vector of all 1s for every sample
            in the case that no weights were passed on upon initialization.
        **kwargs: dict, optional
            If specified, any extra keyword arguments will be passed along to
            the passed ``distribution.pdf()`` function for computing values of
            predicted samples.

        Note: `distribution` should be a frozen distribution if using `scipy`.

        Warnings
        --------
        If passing a `distribution` argument, make sure that the initial
        distribution has been set first, either by having run
        :meth:`set_initial` or :meth:`fit` first.
        """
        if weights is not None:
            self.set_weights(weights)

        if distribution is None:
            # Reweight kde of predicted by weights if present
            distribution = gkde(self.y.T, bw_method=bw_method, weights=self._weights)
            pred_pdf_values = distribution.pdf(self.y.T).T
        else:
            pred_pdf_values = distribution.pdf(self.y, **kwargs)

        self._pr_dist = distribution
        self._pr = pred_pdf_values.ravel()
        self._up = None

    def fit(self, **kwargs):
        """
        Update Initial Distribution

        Constructs the updated distribution by fitting observed data to
        predicted data with:

        .. math::
            \\pi_{up}(\\lambda) = \\pi_{in}(\\lambda)
            \\frac{\\pi_{ob}(Q(\\lambda))}{\\pi_{pred}(Q(\\lambda))}
            :label: data_consistent_solution

        Note that if initial, predicted, and observed distributions have not
        been set before running this method, they will be run with default
        values. To set specific predicted, observed, or initial distributions
        use the ``set_`` methods.

        Parameters
        -----------
        **kwargs : dict, optional
            If specified, optional arguments are passed to the
            :meth:`set_predicted` call in the case that the predicted
            distribution has not been set yet.

        Returns
        -----------

        """
        if self._in is None:
            self.set_initial()
        if self._pr is None:
            self.set_predicted(**kwargs)
        if self._ob is None:
            self.set_observed()

        # Store ratio of observed/predicted
        # e.g. to comptue E(r) and to pass on to future iterations
        self._r = np.divide(self._ob, self._pr)

        # Multiply by initial to get updated pdf
        up_pdf = np.multiply(self._in * self._weights, self._r)
        self._up = up_pdf

    def mud_point(self):
        """Maximal Updated Density (MUD) Point

        Returns the Maximal Updated Density or MUD point as the parameter
        sample from the initial distribution with the highest update density
        value:

        .. math::
            \\lambda^{MUD} := \\mathrm{argmax} \\pi_{up}(\\lambda)
            :label: mud

        Note if the updated distribution has not been computed yet, this
        function will call :meth:`fit` to compute it.

        Parameters
        ----------

        Returns
        -------
        mud_point : np.ndarray
            Maximal Updated Density (MUD) point.
        """
        if self._up is None:
            self.fit()
        m = np.argmax(self._up)
        return self.X[m, :]

    def estimate(self):
        """Estimate

        Returns the best estimate for most likely parameter values for the
        given model data using the data-consistent framework.

        Parameters
        ----------

        Returns
        -------
        mud_point : np.ndarray
            Maximal Updated Density (MUD) point.
        """
        return self.mud_point()

    def expected_ratio(self):
        """Expectation Value of R

        Returns the expectation value of the R, the ratio of the observed to
        the predicted density values.

        .. math::
            R = \\frac{\\pi_{ob}(\\lambda)}
                      {\\pi_{pred}(\\lambda)}
            :label: r_ratio

        If the predictability assumption for the data-consistent framework is
        satisfied, then :math:`E[R]\\approx 1`.

        Parameters
        ----------

        Returns
        -------
        expected_ratio : float
            Value of the E(r). Should be close to 1.0.
        """
        if self._up is None:
            self.fit()

        return np.average(self._r, weights=self._weights)
