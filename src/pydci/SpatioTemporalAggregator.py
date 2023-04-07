import pickle

import numpy as np
from matplotlib import pyplot as plt  # type: ignore

from pydci.MUDProblem import MUDProblem
from pydci.pca import pca, svd
from pydci.utils import add_noise, fit_domain

try:
    import xarray as xr  # type: ignore

    xr_avial = True
except ModuleNotFoundError:
    xr_avail = False
    pass


class SpatioTemporalAggregator(object):
    """
    Class for parameter estimation problems related to spatio-temporal problems.
    equation models of real world systems. Uses a QoI map of weighted
    residuals between simulated data and measurements to do inversion

    Attributes
    ----------
    TODO: Finish

    Methods
    -------
    TODO: Finish


    """

    def __init__(self, df=None):
        self._domain = None
        self._lam = None
        self._data = None
        self._measurements = None
        self._true_lam = None
        self._true_vals = None
        self._sample_dist = None
        self.sensors = None
        self.times = None
        self.qoi = None
        self.pca = None
        self.std_dev = None

        if df is not None:
            self.load(df)

    @property
    def n_samples(self) -> int:
        if self.lam is None:
            raise AttributeError("lambda not yet set.")
        return self.lam.shape[0]

    @property
    def n_qoi(self) -> int:
        if self.qoi is None:
            raise AttributeError("qoi not yet set.")
        return self.qoi.shape[1]

    @property
    def n_sensors(self) -> int:
        if self.sensors is None:
            raise AttributeError("sensors not yet set.")
        return self.sensors.shape[0]

    @property
    def n_ts(self) -> int:
        if self.times is None:
            raise AttributeError("times not yet set.")
        return self.times.shape[0]

    @property
    def n_params(self) -> int:
        return self.domain.shape[0]

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, lam):
        lam = np.array(lam)
        lam = lam.reshape(-1, 1) if lam.ndim == 1 else lam

        if self.domain is not None:
            if lam.shape[1] != self.n_params:
                raise ValueError("Parameter dimensions do not match domain specified.")
        else:
            # TODO: Determine domain from min max in parameters
            self.domain = np.vstack([lam.min(axis=0), lam.max(axis=0)]).T
        if self.sample_dist is None:
            # Assume uniform distribution by default
            self.sample_dist = "u"

        self._lam = lam

    @property
    def lam_ref(self):
        return self._lam_ref

    @lam_ref.setter
    def lam_ref(self, lam_ref):
        if self.domain is None:
            raise AttributeError("domain not yet set.")
        lam_ref = np.reshape(lam_ref, (-1))
        for idx, lam in enumerate(lam_ref):
            if (lam < self.domain[idx][0]) or (lam > self.domain[idx][1]):
                raise ValueError(
                    f"lam_ref at idx {idx} must be inside {self.domain[idx]}."
                )
        self._lam_ref = lam_ref

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        domain = np.reshape(domain, (-1, 2))
        if self.lam is not None:
            if self.domain.shape[0] != self.lam.shape[1]:
                raise ValueError("Domain and parameter array dimension mismatch.")
            min_max = np.vstack([self.lam.min(axis=0), self.lam.max(axis=0)]).T
            if not all(
                [all(domain[:, 0] <= min_max[:, 0]), all(domain[:, 1] >= min_max[:, 1])]
            ):
                raise ValueError("Parameter values exist outside of specified domain")

        self._domain = domain

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        dim = data.shape
        ndim = data.ndim
        if ndim == 1:
            data = np.reshape(data, (-1, 1))
        if ndim == 3:
            # Expected to be in (# samples x # sensors # # timesteps)
            data = np.reshape(data, (dim[0], -1))

        dim = data.shape
        ndim = data.ndim
        if self.sensors is None and self.times is None:
            self.sensors = np.array([0])
            self.times = np.arange(0, dim[1])
        if self.sensors is not None and self.times is None:
            if self.sensors.shape[0] != dim[1]:
                raise ValueError(
                    "Dimensions of simulated data does not match number of sensors"
                )
            self.times = np.array([0])
        if self.sensors is None and self.times is not None:
            if self.times.shape[0] != dim[1]:
                raise ValueError(
                    "Dimensions of simulated data does not match number of timesteps"
                )
            self.sensors = np.array([0])
        if self.sensors is not None and self.times is not None:
            # Assume data is already flattened, check dimensions match
            if self.times.shape[0] * self.sensors.shape[0] != dim[1]:
                raise ValueError(
                    "Dimensions of simulated data != (timesteps x sensors)"
                )

        # Flatten data_data into 2d array
        self._data = data

    @property
    def measurements(self):
        return self._measurements

    @measurements.setter
    def measurements(self, measurements):
        measurements = np.reshape(measurements, (self.n_sensors * self.n_ts, 1))
        self._measurements = measurements

    @property
    def true_vals(self):
        return self._true_vals

    @true_vals.setter
    def true_vals(self, true_vals):
        true_vals = np.reshape(true_vals, (self.n_sensors * self.n_ts, 1))
        self._true_vals = true_vals

    @property
    def sample_dist(self):
        return self._sample_dist

    @sample_dist.setter
    def sample_dist(self, dist):
        if dist not in ["u", "n"]:
            raise ValueError(
                "distribution could not be inferred. Must be from ('u', 'n')"
            )
        self._sample_dist = dist

    def get_closest_to_measurements(
        self,
        samples_mask=None,
        times_mask=None,
        sensors_mask=None,
    ):
        """
        Get closest simulated data point to measured data in $l^2$-norm.
        """
        lam, times, sensors, sub_data, sub_meas = self.sample_data(
            samples_mask=samples_mask,
            times_mask=times_mask,
            sensors_mask=sensors_mask,
        )
        closest_idx = np.argmin(np.linalg.norm(sub_data - sub_meas.T, axis=1))
        closest_lam = lam[closest_idx, :].ravel()

        return closest_lam

    def get_closest_to_true_vals(
        self,
    ):
        """
        Get closest simulated data point to noiseless true values in $l^2$-norm.

        Note for now no sub-sampling implemented here.
        """
        if self.true_vals is None:
            raise AttributeError("True values is not set")
        closest_idx = np.argmin(np.linalg.norm(self.data - self.true_vals.T, axis=1))
        closest_lam = self.lam[closest_idx, :].ravel()

        return closest_lam

    def measurements_from_reference(self, ref=None, std_dev=None, seed=None):
        """
        Add noise to a reference solution.
        """
        if ref is not None:
            self._true_vals = ref
        if std_dev is not None:
            self.std_dev = std_dev
        if self.true_vals is None or self.std_dev is None:
            raise AttributeError(
                "Must set reference solution and std_dev first or pass as arguments."
            )
        self.measurements = np.reshape(
            add_noise(self.true_vals.ravel(), self.std_dev, seed=seed),
            self.true_vals.shape,
        )

    def load(
        self,
        df,
        lam="lam",
        data="data",
        **kwargs,
    ):
        """
        Load data from a file on disk for a PDE parameter estimation problem.

        Parameters
        ----------
        fname : str
            Name of file on disk. If ends in '.nc' then assumed to be netcdf
            file and the xarray library is used to load it. Otherwise the
            data is assumed to be pickled data.

        Returns
        -------
        ds : dict,
            Dictionary containing data from file for PDE problem class

        """
        if type(df) == str:
            try:
                if df.endswith("nc") and xr_avail:
                    ds = xr.load_dataset(df)
                else:
                    with open(df, "rb") as fp:
                        ds = pickle.load(fp)
            except FileNotFoundError:
                raise FileNotFoundError(f"Couldn't find data file {df}")
        else:
            ds = df

        def get_set_val(f, v):
            if f in ds.keys():
                self.__setattr__(f, ds[v])
            elif v is not None and type(v) != str:
                self.__setattr__(f, v)

        field_names = {
            "sample_dist": "sample_dist",
            "domain": "domain",
            "sensors": "sensors",
            "times": "times",
            "lam_ref": "lam_ref",
            "std_dev": "std_dev",
            "true_vals": "true_vals",
            "measurements": "measurements",
        }
        field_names.update(kwargs)
        for f, v in field_names.items():
            get_set_val(f, v)

        get_set_val("lam", lam)
        get_set_val("data", data)

        return ds

    def validate(
        self,
        check_meas=True,
        check_true=False,
    ):
        """Validates if class has been set-up appropriately for inversion"""
        req_attrs = ["domain", "lam", "data"]
        if check_meas:
            req_attrs.append("measurements")
        if check_true:
            req_attrs.append("true_lam")
            req_attrs.append("true_vals")

        missing = [x for x in req_attrs if self.__getattribute__(x) is None]
        if len(missing) > 0:
            raise ValueError(f"Missing attributes {missing}")

    def sample_data(
        self,
        samples_mask=None,
        times_mask=None,
        sensors_mask=None,
    ):
        if self.data is None:
            raise AttributeError("data not set yet.")

        sub_data = np.reshape(self.data, (self.n_samples, self.n_sensors, self.n_ts))
        if self.measurements is not None:
            sub_meas = np.reshape(self.measurements, (self.n_sensors, self.n_ts))
        else:
            sub_meas = None

        sub_times = self.times
        sub_sensors = self.sensors
        sub_lam = self.lam
        if samples_mask is not None:
            sub_lam = self.lam[samples_mask, :]
            sub_data = sub_data[samples_mask, :, :]
        if times_mask is not None:
            sub_times = np.reshape(self.times[times_mask], (-1, 1))
            sub_data = sub_data[:, :, times_mask]
            if sub_meas is not None:
                sub_meas = sub_meas[:, times_mask]
        if sensors_mask is not None:
            sub_sensors = np.reshape(self.sensors[sensors_mask], (-1, 2))
            sub_data = sub_data[:, sensors_mask, :]
            if sub_meas is not None:
                sub_meas = sub_meas[sensors_mask, :]

        sub_data = np.reshape(sub_data, (-1, sub_times.shape[0] * sub_sensors.shape[0]))
        if sub_meas is not None:
            sub_meas = np.reshape(sub_meas, (len(sub_times) * len(sub_sensors)))

        return sub_lam, sub_times, sub_sensors, sub_data, sub_meas

    def sensor_contour_plot(
        self, idx=0, c_vals=None, ax=None, mask=None, fill=True, colorbar=True, **kwargs
    ):
        """
        Plot locations of sensors in space
        """
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)

        sensors = self.sensors[mask, :] if mask is not None else self.sensors
        contours = c_vals if c_vals is not None else self.data[idx, :].ravel()

        if fill:
            tc = ax.tricontourf(sensors[:, 0], sensors[:, 1], contours, **kwargs)
        else:
            tc = ax.tricontour(sensors[:, 0], sensors[:, 1], contours, **kwargs)

        if colorbar:
            fig = plt.gcf()
            fig.colorbar(tc)

        ax.set_aspect("equal")

        return ax

    def sensor_scatter_plot(self, ax=None, mask=None, colorbar=None, **kwargs):
        """
        Plot locations of sensors in space
        """
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)

        sensors = self.sensors[mask, :] if mask is not None else self.sensors

        sp = plt.scatter(sensors[:, 0], sensors[:, 1], **kwargs)

        if colorbar and "c" in kwargs.keys():
            fig = plt.gcf()
            fig.colorbar(sp)

        return ax

    def plot_ts(
        self,
        ax=None,
        samples=None,
        times=None,
        sensor_idx=0,
        max_plot=100,
        meas_kwargs={},
        samples_kwargs={},
        alpha=0.1,
    ):
        """
        Plot time series data
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(1, 1, 1)

        if self.data is None and self.measurements is None:
            raise ValueError("No data to plot")

        lam, times, _, sub_data, sub_meas = self.sample_data(
            samples_mask=samples, times_mask=times, sensors_mask=sensor_idx
        )

        num_samples = sub_data.shape[0]
        if sub_meas is not None:
            max_plot = num_samples if max_plot > num_samples else max_plot

            # Plot measured time series
            def_kwargs = {
                "color": "k",
                "marker": "^",
                "zorder": 50,
                "s": 2,
            }
            def_kwargs.update(meas_kwargs)
            _ = plt.scatter(times, sub_meas, **def_kwargs)

        # Plot simulated data time series
        def_sample_kwargs = {"color": "r", "linestyle": "-", "zorder": 1, "alpha": 0.1}
        def_sample_kwargs.update(samples_kwargs)
        for i, idx in enumerate(np.random.choice(num_samples, max_plot)):
            if i != (max_plot - 1):
                _ = ax.plot(times, sub_data[i, :], **def_sample_kwargs)
            else:
                _ = ax.plot(
                    times,
                    sub_data[i, :],
                    "r-",
                    alpha=alpha,
                    label=f"Sensor {sensor_idx}",
                )

        return ax

    def mud_problem(
        self,
        method="pca",
        data_weights=None,
        sample_weights=None,
        num_components=2,
        samples_mask=None,
        times_mask=None,
        sensors_mask=None,
    ):
        """Build QoI Map Using Data and Measurements"""

        # TODO: Finish sample data implementation
        lam, times, sensors, sub_data, sub_meas = self.sample_data(
            samples_mask=samples_mask,
            times_mask=times_mask,
            sensors_mask=sensors_mask,
        )
        residuals = (sub_meas - sub_data) / self.std_dev
        sub_n_samples = sub_data.shape[0]

        if data_weights is not None:
            data_weights = np.reshape(data_weights, (-1, 1))
            if data_weights.shape[0] != self.n_sensors * self.n_ts:
                raise ValueError(
                    "Data weights vector and dimension of data space does not match"
                )
            data_weights = data_weights / np.linalg.norm(data_weights)
            residuals = data_weights * residuals

        if method == "wme":
            qoi = np.sum(residuals, axis=1) / np.sqrt(sub_n_samples)
        elif method == "pca":
            # Learn qoi to use using PCA
            pca_res, X_train = pca(residuals, n_components=num_components)
            self.pca = {
                "X_train": X_train,
                "vecs": pca_res.components_,
                "var": pca_res.explained_variance_,
            }

            # Compute WME
            qoi = residuals @ pca_res.components_.T
        elif method == "svd":
            # Learn qoi to use using SVD
            u, s, v = svd(residuals)
            self.svd = {"u": u, "singular_values": s, "singular_vectors": v}
            qoi = residuals @ (v[0:num_components, :]).T
        else:
            ValueError(f"Unrecognized QoI Map type {method}")

        # qoi = qoi.reshape(sub_n_samples, -1)
        d = MUDProblem(lam, qoi, self.domain, weights=sample_weights)

        return d
