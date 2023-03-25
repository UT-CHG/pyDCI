from pathlib import Path
import pdb
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista
import ufl
from alive_progress import alive_bar
from dafi import random_field as rf
from dolfinx import fem, io, mesh, plot
from mpi4py import MPI
from petsc4py import PETSc


class HeatModel:
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
        T=1.0,
        t=0.0,
        dt=0.001,
        sample_ts=0.2,
        nx=50,
        ny=50,
        mean=0.0,
        std_dev=1.0,
        length_scales=[0.1, 0.1],
        nmodes=10,
        true_k_x=None,
        init_cond=None,
        setup=True,
    ):
        self.T = T
        self.t = t
        self.dt = dt
        self.num_steps = int(T / dt)
        self.sample_ts = sample_ts
        self.nx = nx
        self.ny = ny
        self.mean = mean
        self.sd = std_dev
        self.lscales = length_scales
        self.nmodes = nmodes
        self.true_k_x = true_k_x

        # Create initial condition
        def def_init(x, a=5):
            return np.exp(-a * (x[0] ** 2 + x[1] ** 2))

        self.initial_condition = def_init if init_cond is None else init_cond

        # Setup simulation
        if setup:
            self.setup_simulation()

    def _create_variational_problem(self, params=None):
        """
        Build Variational Problem

        This is called on each run of `run_model()` to reset the parameters
        of the simulation as necessary.
        """
        if params is not None:
            self.set_k_x(params)

        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        f = fem.Constant(self.domain, PETSc.ScalarType(0))
        a = (
            u * v * ufl.dx
            + self.dt * self.kx * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        L = (self.u_n + self.dt * f) * v * ufl.dx

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

    def set_kl(self, lscales=None, nmodes=None, sd=None, normalize=True):
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
        self.modes = rf.calc_kl_modes(self.cov, nmodes=self.nmodes, normalize=normalize)

        # print(f"Naive Evalues: {self.modes[0].shape}")
        # print(f"Naive Modes: {self.modes[1].shape}")

        # self.evalues1, self.modes1 = rf.calc_kl_modes_coverage(
        #     self.cov, coverage=0.99, normalize=normalize
        # )
        # print(f"Coverage Evalues: {self.evalues1.shape}")
        # print(f"Coverage Modes: {self.modes1.shape}")

    def set_mean(self, mean=None):
        """
        Seat mean of KL expansion
        """
        if mean is not None:
            self.mean = mean

        # Work k_x into a KL expansion representation
        if isinstance(self.mean, float) or isinstance(self.mean, int):
            self.mean = float(self.mean) * np.ones(self.coords[:, 0].shape)
        elif isinstance(self.mean, Callable):
            self.mean = self.mean(self.coords[:, 0], self.coords[:, 1])

    def set_k_x(self, k_x=None):
        """
        Set thermal diffusivity k_x function over space
        """
        if k_x is None:
            k_x = self.true_k_x

        # make sure mean of KL expansion set
        self.set_mean()

        # Work k_x into a KL expansion representation
        if isinstance(k_x, float) or isinstance(k_x, int):
            # Constant over the space
            value = k_x

            def default_k_x(x):
                return value * np.ones(len(x[0]))

            k_x = default_k_x

        if isinstance(k_x, Callable):
            # Project function onto space by evaluating over field
            _, k_x, _ = self.project_field(k_x)

        if k_x is None:
            # None for k_x is
            k_x = np.random.normal(0, 1, [1, self.nmodes])[0]

        self.kx = fem.Function(self.V)
        self.kx.x.array[:] = rf.reconstruct_kl(self.modes[1], k_x, mean=self.mean)[:, 0]

        return k_x

    def project_field(self, field_fun):
        """
        Given a python function f(x, y) -> z, project function using
        KL expansion intiialized in init_field()
        """
        vals = field_fun([self.coords[:, 0], self.coords[:, 1]])
        proj = rf.project_kl(vals, self.modes[1], mean=self.mean)
        proj_vals = rf.reconstruct_kl(self.modes[1], proj, mean=self.mean)
        return vals, proj, proj_vals

    def setup_simulation(self):
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
        self.set_kl(lscales=self.lscales, nmodes=self.nmodes, sd=self.sd)

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

        # Initialize thermal diffusivity
        self.true_params = self.set_k_x()

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

        for i in range(self.num_steps):
            self.t += self.dt
            snap_counter += self.dt

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

    def reset_sim(self):
        """
        Reset a model run

        Resets the current time of the HeatModel instance to 0, and
        interpolates the initial condition to the function spaces u_n and uh.
        Call after run_model() if you want to re-run with different params.
        """
        self.t = 0
        self.u_n.interpolate(self.initial_condition)
        self.uh.interpolate(self.initial_condition)

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
            field = rf.reconstruct_kl(self.modes[1], self.true_params, mean=self.mean)
        elif isinstance(field, Callable):
            if project:
                field, _, _ = self.project_field(field)
            else:
                field = field([self.coords[:, 0], self.coords[:, 1]])
        else:
            if field.shape == self.true_params.shape:
                field = rf.reconstruct_kl(self.modes[1], field, mean=self.mean)

        return field

    def plot_field(self, field=None, project=False, cbar=True, diff=None, **kwargs):
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

        sc = ax.scatter(self.coords[:, 0], self.coords[:, 1], c=field, cmap="seismic")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Field Sample $k(\mathbf{x})$")

        if cbar:
            cbar = plt.colorbar(sc)
            cbar.set_label("F(x)")

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
                self.reset_sim()
                all_res.append(self.run_model(param))
                bar()

        sample_keys = [f"s_{x}" for x in range(len(params))]
        all_res = pd.concat(all_res, keys=sample_keys)

        if out_file is not None:
            all_res.to_csv(out_file, index=True)
            return params, out_file
        else:
            return params, all_res


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
