import pdb
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ufl
from alive_progress import alive_bar
from dafi import random_field as rf
from dolfinx import fem, io, mesh, plot
from mpi4py import MPI
from petsc4py import PETSc
from scipy.stats.distributions import norm, uniform

from pydci.Model import DynamicModel
from pydci.log import logger

pyvsita_flag = False
try:
    import pyvista
    pyvsita_flag = True
except ImportError as ie:
    logger.warning('Pyvista not found')



class HeatModel(DynamicModel):
    """
    Solves the Heat Equation using Finite Element Method with Fenics.


    Attributes
    ----------
    T : float, optional
        Final time. Default is 1.0.
    t : float, optional
        Start time. Default is 0.0.
    dt : float, optional
        Time step size. Default is 0.001.
    sample_ts : float, optional
        Time interval for saving snapshots of solution. Default is 0.2.
    nx : int, optional
        Number of degrees of freedom in the x direction. Default is 50.
    ny : int, optional
        Number of degrees of freedom in the y direction. Default is 50.
    mean : float, optional
        Mean value for the Gaussian Process. Default is 0.0.
    std_dev : float, optional
        Standard deviation value for the Gaussian Process. Default is 1.0.
    length_scales : list of floats, optional
        Length scales for the kernel of the Gaussian Process. Default is [0.1, 0.1].
    nmodes : int, optional
        Number of modes for the solution. Default is 10.
    true_k_x : None or callable, optional
        A function representing the true diffusion coefficient. Default is None.
    """

    def __init__(
        self,
        x0=None,
        t0=0.0,
        measurement_noise=0.05,
        solve_ts=0.0001,
        sample_ts=0.1,
        nx=50,
        ny=50,
        mean=1.0,
        std_dev=1.0,
        length_scales=[0.1, 0.1],
        nmodes=4,
        true_k_x=None,
        max_states=500,
    ):
        self.nx = nx
        self.ny = ny
        self.mean = mean
        self.sd = std_dev
        self.lscales = length_scales
        self.nmodes = nmodes
        # Create initial condition
        def def_init(x, a=5):
            return np.exp(-a * (x[0] ** 2 + x[1] ** 2))
        self.initial_condition = def_init if x0 is None else x0

        # Setup simulation
        self.setup_simulation(true_k_x=true_k_x)

        # Note we set a dummy lam_true for now because when we set thermal
        # diffusivity field we overwrite it with the true lam coefficients.
        super().__init__(
            self.uh.x.array.ravel(),
            self.lam_true,
            t0=t0,
            measurement_noise=measurement_noise,
            solve_ts=solve_ts,
            sample_ts=sample_ts,
            param_mins=None,
            param_maxs=None,
            param_shifts=None,
            max_states=max_states,
        )

    def _create_variational_problem(self, params=None):
        """
        Build Variational Problem

        This is called on each run of `run_model()` to reset the parameters
        of the simulation as necessary.
        """
        params = self.lam_ture if params is None else params
        proj_vals = self.reconstruct(params, log=True)
        kx = fem.Function(self.V)
        kx.x.array[:] = proj_vals

        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        f = fem.Constant(self.domain, PETSc.ScalarType(0))
        a = (
            u * v * ufl.dx
            + self.solve_ts * kx * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        L = (self.u_n + self.solve_ts * f) * v * ufl.dx

        # Prepare linear algebra objects
        self.bilinear_form = fem.form(a)
        self.linear_form = fem.form(L)

        A = fem.petsc.assemble_matrix(self.bilinear_form, bcs=[self.boundary_condition])
        A.assemble()
        self.b = fem.petsc.create_vector(self.linear_form)

        # Petsc solver
        solver = PETSc.KSP().create(self.domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver = solver

    def project(self, field=None, mean=None, log=True):
        """
        Set thermal diffusivity field function over space
        """
        if field is None:
            field = self.true_field

        # make sure mean of KL expansion set
        if mean is not None:
            self.mean = mean

        # Work field into a KL expansion representation
        if isinstance(self.mean, float) or isinstance(self.mean, int):
            self.mean = float(self.mean) * np.ones(self.coords[:, 0].shape)
        elif isinstance(self.mean, Callable):
            self.mean = self.mean(self.coords[:, 0], self.coords[:, 1])

        # Work field into a KL expansion representation
        if isinstance(field, float) or isinstance(field, int):
            # Constant over the space
            field_vals = field * np.ones(len(self.coords[:, 0]))
        elif isinstance(field, Callable):
            # Project function onto space by evaluating over field
            field_vals = field([self.coords[:, 0], self.coords[:, 1]])
        else:
            raise ValueError('field must be either a float/int for constant' + \
                            'over domain or a function, an array of KL ' + \
                            'coeffiecients or None for random field of' + \
                            f'coeffiecients. Type: {type(field)}')

        if log:
            log_vals = np.log(field_vals / mean)
            return rf.project_kl(log_vals, self.modes[1], mean=np.log(mean))
        else:
            return rf.project_kl(field_vals, self.modes[1], mean=mean)

    def setup_simulation(self, true_k_x=None):
        """
        Setup Fenics Simulaltion domain, boundary, and functions.
        """
        # Define Domain
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([-2, -2]), np.array([2, 2])],
            [self.nx, self.ny],
            mesh.CellType.triangle,
        )

        # Function space
        self.V = fem.FunctionSpace(self.domain, ("CG", 1))
        self.coords = self.V.tabulate_dof_coordinates()[:, :2]
        self.init_kl(lscales=self.lscales, nmodes=self.nmodes, sd=self.sd)

        # Initial Conditions
        u_n = fem.Function(self.V)
        u_n.name = "u_n"
        u_n.interpolate(self.initial_condition)
        self.u_n = u_n

        # Create boundary condition
        fdim = self.domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        self.boundary_condition = fem.dirichletbc(
            PETSc.ScalarType(0),
            fem.locate_dofs_topological(self.V, fdim, boundary_facets),
            self.V,
        )

        uh = fem.Function(self.V)
        uh.name = "uh"
        uh.interpolate(self.initial_condition)
        self.uh = uh

        # Initialize thermal diffusivity - Project true_k_x onto KL field,
        # Then reconstruct and set to kx array for variatonal prob
        if true_k_x is None:
            self.lam_true = np.random.normal(0, 1, [1, self.nmodes])[0]
        else:
            self.lam_true = self.project(field=true_k_x,
                                         mean=self.mean, log=True)

    def init_kl(self, lscales=None, nmodes=None, sd=None, normalize=False):
        """
        Initializes KL Decomposition over grid for building thermal diffusivity
        field. Note the KL modes used at initialization are used throughout the
        whole lifecycle of this class.

        Parameters
        ----------
        lscales : list of floats, optional
            Length scales for the kernel of the Gaussian Process. If None, the
            length scales specified in the HeatModel instance will be used.
            Default is None.
        nmodes : int, optional
            Number of modes for the KL decomposition. If None, the number of
            modes specified in the HeatModel instance will be used. Default is
            None.
        sd : float, optional
            Standard deviation value for the Gaussian Process. If None, the
            standard deviation specified at initialization will be used.
            Default is None.
        normalize : bool, optional
            Whether to normalize the KL modes.
        """
        if lscales is not None:
            self.lscales = lscales
        if nmodes is not None:
            self.nmodes = nmodes
        if sd is not None:
            self.sd = sd

        n_points, n_dims = self.coords.shape
        exp = np.zeros([n_points, n_points])
        for i in range(2):
            x_1, x_2 = np.meshgrid(self.coords[:, i], self.coords[:, i])
            exp += ((x_1 - x_2) / (self.lscales[i])) ** 2.0

        self.cov = self.sd**2 * np.exp(-0.5 * exp)
        self.modes = rf.calc_kl_modes(self.cov,
                                      nmodes=self.nmodes,
                                      normalize=normalize)

    def reconstruct(self, projection, mean=None, log=True):
        """
        Given a set of kl coefficients
        """
        mean = self.mean if mean is None else mean
        if log:
            mean = np.log(self.mean)
            log_proj_vals = rf.reconstruct_kl(self.modes[1],
                                              projection,
                                              mean=mean)
            return self.mean * np.exp(log_proj_vals)[:,0]
        else:
            return rf.reconstruct_kl(self.modes[1], projection, mean=mean)

    def run_model(self, params=None, fname=None, sample_ts=None):
        """
        Run the model with the given set of parameters and return snapshots of
        the solution as a Pandas DataFrame.

        Parameters
        ----------
        params : ndarray, optional
            Parameters to use for thermal diffusivity k_x
        fname : str, optional
            Name of file to save the snapshots as an XDMF file.
        sample_ts : float, optional
            Time interval between snapshots.

        Returns
        -------
        pandas.DataFrame
            Concatenated Pandas DataFrame of snapshots of the solution.

        Raises
        ------
        RuntimeError
            If `params` is not provided.

        Example
        -------
        >>> from HeatModel import HeatModel
        >>> import pandas as pd
        >>> model = HeatModel()
        >>> snapshots = model.run_model()
        >>> isinstance(snapshots, pd.DataFrame)
        True
        """
        self._create_variational_problem(params)

        if fname is not None:
            xdmf = io.XDMFFile(self.domain.comm, fname, "w")
            xdmf.write_mesh(self.domain)

        snapshots = []
        sample_ts = self.sample_ts if sample_ts is None else sample_ts

        def take_snap(u):
            snapshots.append(
                pd.DataFrame(
                    np.array(u).T,
                    columns=[f"t_{len(snapshots)}"],
                )
            )

        take_snap(self.uh.x.array.copy())
        snap_counter = 0.0

        num_steps = int(self.T / self.solve_ts)
        for i in range(num_steps):
            self.t += self.solve_ts
            snap_counter += self.solve_ts

            # Update the right hand side reusing the initial vector
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b, self.linear_form)

            # Apply Dirichlet boundary condition to the vector
            fem.petsc.apply_lifting(
                self.b, [self.bilinear_form], [[self.boundary_condition]]
            )
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            fem.petsc.set_bc(self.b, [self.boundary_condition])

            # Solve linear problem
            self.solver.solve(self.b, self.uh.vector)
            self.uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            self.u_n.x.array[:] = self.uh.x.array
            if snap_counter >= sample_ts:
                take_snap(self.uh.x.array.copy())
                snap_counter = 0.0
                if fname is not None:
                    xdmf.write_function(self.uh, self.t)

        snapshots = pd.concat(snapshots, axis=1)

        if fname is not None:
            xdmf.close()

        return snapshots

    def _init_axis(self, **kwargs):
        """
        Plotting utility for initializing figures
        """
        if "ax" in kwargs.keys():
            ax = kwargs.pop("ax")
        else:
            _, ax = plt.subplots(1, 1, figsize=(4, 4))

        return ax, kwargs

    def _process_field(self, field, project=False):
        """ """
        if field is None:
            field = self.reconstruct(self.lam_true)
        elif isinstance(field, Callable):
            if project:
                proj = self.project(field)
                field = self.reconstruct(proj)
            else:
                field = field([self.coords[:, 0], self.coords[:, 1]])
        else:
            if field.shape == self.lam_true.shape:
                field = self.reconstruct(field)

        return field

    def plot_field(self,
                   field=None,
                   project=False,
                   cbar=True,
                   diff=None,
                   **kwargs):
        """
        Plot a field characterized by either:
          1. function - field is a callable on the coordinates array. This
          function is evaluated on the grid, projected onto the KL expansion,
          and plotted.
          2. An array - field values as a matrix over the coordinate array.
          This is projected onto KL expansion and plotted.
          3. Parameter array - Coefficients of KL modes to expand.
        If nothing is passed in the field argument, the true field as stored by
        on initialization is plotted.
        """
        ax, kwargs = self._init_axis(**kwargs)

        field = self._process_field(field, project=project)
        if diff is not None:
            field = field - self._process_field(diff, project=project)

        sc = ax.scatter(self.coords[:, 0], self.coords[:, 1],
                        c=field, cmap="seismic")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Field Sample $k(\mathbf{x})$")

        if cbar:
            cbar = plt.colorbar(sc)
            cbar.set_label("F(x)")

    def plot_kl_diff(self, field=None, **kwargs):
        """
        Plot a difference between a field over the space and its KL expansion.
        """
        self.plot_field(field, diff=self.project(field))

    def plot_solution(self, snapshots, idx=0, cbar=True, **kwargs):
        """
        Quck scatter plot of a snapshot of a solution

        Parameters
        ----------
        snapshots : pandas dataframe
            Result of `run_model()`. A dataframe with columns 't_{i}' for
            i = 1 to T timesteps, where T is the number snap shots taken
            during the simulation. Each row of the dataframe contains the
            value of the solution at the corresponding index point in the grid
            at that time.
        idx: int, optional
            Time step to plot solution for. Default = 0.
        """
        ax, kwargs = self._init_axis(**kwargs)
        sc = ax.scatter(
            self.coords[:, 0], self.coords[:, 1], c=snapshots[f"t_{idx}"], **kwargs
        )

        if cbar:
            cbar = plt.colorbar(sc)
            cbar.set_label("u(x)")

    def output_gif(self, snapshots, gif_name="u_time.gif", diff=None):
        """
        Plot the solution over time in a gif using pyvista.

        Parameters
        ----------
        snapshots : pandas dataframe
            Result of `run_model()`. A dataframe with columns 't_{i}' for
            i = 1 to T timesteps, where T is the number snap shots taken
            during the simulation. Each row of the dataframe contains the
            value of the solution at the corresponding index point in the grid
            at that time.
        gif_name : str, optional
            The name of the output gif file. Default is 'u_time.gif'.
        """
        if not pyvista_flag:
            raise ImportError('Cannot create gif - pyvista not found')
        grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(self.V))
        plotter = pyvista.Plotter()
        plotter.open_gif(gif_name)
        to_plot = snapshots if diff is None else snapshots - diff
        grid.point_data["uh"] = to_plot["t_0"]
        warped = grid.warp_by_scalar("uh", factor=1)
        viridis = plt.cm.get_cmap("viridis", 25)
        sargs = dict(
            title_font_size=25,
            label_font_size=20,
            fmt="%.2e",
            color="black",
            position_x=0.1,
            position_y=0.8,
            width=0.8,
            height=0.1,
        )
        plotter.add_mesh(
            warped,
            show_edges=True,
            lighting=False,
            cmap=viridis,
            scalar_bar_args=sargs,
            clim=[to_plot.min().min(), to_plot.max().max()],
        )
        with alive_bar(to_plot.shape[1], force_tty=True) as bar:
            for i in range(1, to_plot.shape[1]):
                warped = grid.warp_by_scalar("uh", factor=1)
                plotter.update_coordinates(warped.points.copy(), render=False)
                plotter.update_scalars(to_plot[f"t_{i}"], render=False)
                plotter.write_frame()
                bar()

        plotter.close()

    def run_samples(self, nsamples=1, params=None, out_file=None):
        """
        Run parameter samples

        Given a model configuration dictionary, initializes a HeatModel instance with the
        specified parameters, sets up the simulation, generates a data-set for parameter
        estimation by running the model with specified parameters or randomly generated ones.

        Parameters
        ----------
        model_config : dict
            Dictionary containing the parameters for initializing a HeatModel instance.
        nsamples : int, optional
            Number of samples to generate for the data-set. Default is 1.
        params : numpy.ndarray or list of numpy.ndarray, optional
            Array of shape (nsamples, heat_model.nmodes) specifying the parameters to use
            for running the model. If None, random parameters are generated. If a list of
            numpy.ndarray is provided, each array must have the shape (heat_model.nmodes,),
            and the corresponding model runs are performed with each set of parameters.
            Default is None.
        out_file : str, optional
            File path for saving the output data-set as a CSV file. If None, the output is
            returned as a pandas DataFrame. Default is None.

        Returns
        -------
        heat_model : HeatModel
            The initialized HeatModel instance.
        params : numpy.ndarray or list of numpy.ndarray
            The randomly generated or provided parameters used for running the model.
        all_res : pandas.DataFrame or str
            The resulting data-set as a pandas DataFrame or the file path to the saved CSV file.
        """
        if params is None:
            params = np.random.normal(0, 1, [nsamples, self.nmodes])
        else:
            params = [params] if not isinstance(params, list) else params

        all_res = []
        with alive_bar(len(params), force_tty=True) as bar:
            for param in params:
                self.t = 0
                self.u_n.interpolate(initial_condition)
                self.uh.interpolate(initial_condition)
                all_res.append(self.run_model(param))
                bar()

        sample_keys = [f"s_{x}" for x in range(len(params))]
        all_res = pd.concat(all_res, keys=sample_keys)

        if out_file is not None:
            all_res.to_csv(out_file, index=True)
            return params, out_file
        else:
            return params, all_res

    def forward_model(
        self,
        x0: List[float],
        times: np.ndarray,
        lam: np.ndarray,
        fname=None,
    ) -> np.ndarray:
        """
        Forward Model

        Stubb meant to be overwritten by inherited classes.

        Parameters
        ----------
        x0 : List[float]
            Initial conditions.
        times: np.ndarray[float]
            Time steps to solve the model for. Note that times[0] the model
            is assumed to be at state x0.
        parmaeters: Tuple
            Tuple of parameters to set for model run. These should correspond
            to the model parameters being varied.
        """
        # Set initial conditions (how to do?)
        self.u_n.x.array[:] = np.array(x0).ravel()
        self.uh.x.array[:] = np.array(x0).ravel()

        # Create variational problme to solve given thermal diff. field lam
        self._create_variational_problem(np.array(lam))

        sol = np.zeros((len(times), self.n_states))
        for i, t in enumerate(times):
            sol[i] = self.uh.x.array.copy().ravel()

            # Update the right hand side reusing the initial vector
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b, self.linear_form)

            # Apply Dirichlet boundary condition to the vector
            fem.petsc.apply_lifting(
                self.b, [self.bilinear_form], [[self.boundary_condition]]
            )
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            fem.petsc.set_bc(self.b, [self.boundary_condition])

            # Solve linear problem
            self.solver.solve(self.b, self.uh.vector)
            self.uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            self.u_n.x.array[:] = self.uh.x.array

        return sol
    domain.sort(axis=1)

    def k_x_mud_plot(self,
                     iteration=0,
                     figsize=(18,5)):
        """
        Plot estimated and True k(x)
        """
        iteration = 0
        fig, ax = plt.subplots(1, 3, figsize=(18,5))
        self.plot_field(field=self.probs[iteration].mud_point,
                        ax=ax[0])
        ax[0].set_title('$k^{MUD}(x)$')
        self.plot_field(field=self.lam_true, ax=ax[1])
        ax[1].set_title('$k^{\dagger}(x)$')
        ax[1].set_ylabel('')
        self.plot_field(field=self.probs[iteration].mud_point,
                        diff=self.lam_true, ax=ax[2])
        ax[2].set_title('Error')
        ax[2].set_ylabel('')
        fig.tight_layout


# def parallel_run(model_configs, num_samples=10, workers=4):
#     """
#     """
#     res = []
#     with alive_bar(len(param_samples), force_tty=True) as bar:
#         with concurrent.futures.ThreadPoolExecutor(
#             max_workers=workers) as executor:
#             futures = []
#             for idx, m in enumerate(model_configs):
#                 futures.append(executor.submit(
#                   setup_and_run, m, num_samples, idx))
#             for future in concurrent.futures.as_completed(futures):
#                 fname = future.result()
#                 data = pd.read_csv(fname, index_col=False)
#                 data['index'] = data.index
#                 res.append(data)
#                 Path(fname).unlink()
#                 bar()
#
#     res = pd.concat(res, keys=[f's_{x}' for x in range(num_samples)])
#
#     return res
#    def setup_and_run(model_config, nsamples=1, params=None, out_file=None):
#        """
#        Setup and run
#
#        Given a model configuration dictionary, initializes a HeatModel instance with the
#        specified parameters, sets up the simulation, generates a data-set for parameter
#        estimation by running the model with specified parameters or randomly generated ones.
#
#        Parameters
#        ----------
#        model_config : dict
#            Dictionary containing the parameters for initializing a HeatModel instance.
#        nsamples : int, optional
#            Number of samples to generate for the data-set. Default is 1.
#        params : numpy.ndarray or list of numpy.ndarray, optional
#            Array of shape (nsamples, heat_model.nmodes) specifying the parameters to use
#            for running the model. If None, random parameters are generated. If a list of
#            numpy.ndarray is provided, each array must have the shape (heat_model.nmodes,),
#            and the corresponding model runs are performed with each set of parameters.
#            Default is None.
#        out_file : str, optional
#            File path for saving the output data-set as a CSV file. If None, the output is
#            returned as a pandas DataFrame. Default is None.
#
#        Returns
#        -------
#        heat_model : HeatModel
#            The initialized HeatModel instance.
#        params : numpy.ndarray or list of numpy.ndarray
#            The randomly generated or provided parameters used for running the model.
#        all_res : pandas.DataFrame or str
#            The resulting data-set as a pandas DataFrame or the file path to the saved CSV file.
#        """
#        heat_model = HeatModel(**model_config)
#        heat_model.setup_simulation()
#        if params is None:
#            params = np.random.normal(0, 1, [nsamples, heat_model.nmodes])
#        else:
#            params = [params] if not isinstance(params, list) else params
#
#        all_res = []
#        with alive_bar(len(params), force_tty=True) as bar:
#            for param in params:
#                heat_model.reset_sim()
#                all_res.append(heat_model.run_model(param))
#                bar()
#
#        all_res = pd.concat(all_res, keys=[f"s_{x}" for x in range(len(params))])
#
#        if out_file is not None:
#            all_res.to_csv(out_file, index=True)
#            return heat_model, params, out_file
#        else:
#            return heat_model, params, all_res
