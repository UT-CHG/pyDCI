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

from pydci.log import logger
from pydci.Model import DynamicModel

pyvista_flag = False
try:
    import pyvista

    pyvista_flag = True
except ImportError as ie:
    logger.warning("Pyvista not found")


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
        forcing_expression=None,
        model_file=None,
    ):
        if model_file is not None:
            self.load(model_file)

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
        self.forcing_expression = forcing_expression

        # Setup simulation - If loading from file lam_true is already set here.
        if 'lam_true' in self.__dict__.keys():
            logger.debug(f'Setting true_k_x to {self.lam_true} of type {type(self.lam_true)}')
            true_k_x = self.lam_true
        logger.debug("Setting up simulation")
        self.setup_simulation(true_k_x=true_k_x)

        # Set hard-coded value for max number of states to use.
        self.MAX_STATES = max_states

        # Note we set a dummy lam_true for now because when we set thermal
        # diffusivity field we overwrite it with the true lam coefficients.
        super().__init__(
            self.uh.x.array.ravel(),
            self.lam_true,
            measurement_noise=measurement_noise,
            solve_ts=solve_ts,
            sample_ts=sample_ts,
            param_mins=None,
            param_maxs=None,
            param_shifts=None,
        )

    def project(self, field=None, mean=None, log=True):
        """
        Set thermal diffusivity field function over space
        """
        if field is None:
            field = self.lam_true
        if mean is not None:
            self.mean = mean

        # * Mean can be defined as function over coordiante grid or constant over whole grid.
        if isinstance(self.mean, float) or isinstance(self.mean, int):
            self.mean = float(self.mean) * np.ones(self.coords[:, 0].shape)
        elif isinstance(self.mean, Callable):
            self.mean = self.mean(self.coords[:, 0], self.coords[:, 1])

        # * Field can either be a constant over coordinate grid, constant, or ndarray
        if isinstance(field, float) or isinstance(field, int):
            # Constant over the space
            field_vals = field * np.ones(len(self.coords[:, 0]))
        elif isinstance(field, Callable):
            # Project function onto space by evaluating over field
            field_vals = field([self.coords[:, 0], self.coords[:, 1]])
        elif not isinstance(field, np.ndarray):
            raise ValueError(
                "field must be either a float/int for constant"
                + "over domain or a function, an array of KL "
                + "coeffiecients or None for random field of"
                + f"coeffiecients. Type: {type(field)}"
            )

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
            if isinstance(true_k_x, np.ndarray) and true_k_x.shape[0] == self.nmodes:
                self.lam_true = true_k_x
            else:
                self.lam_true = self.project(field=true_k_x, mean=self.mean, log=True)

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
        self.modes = rf.calc_kl_modes(self.cov, nmodes=self.nmodes, normalize=normalize)

    def reconstruct(self, projection, mean=None, log=True):
        """
        Given a set of kl coefficients
        """
        mean = self.mean if mean is None else mean
        if log:
            mean = np.log(self.mean)
            log_proj_vals = rf.reconstruct_kl(self.modes[1], projection, mean=mean)
            return self.mean * np.exp(log_proj_vals)[:, 0]
        else:
            return rf.reconstruct_kl(self.modes[1], projection, mean=mean)

    def _create_variational_problem(self, params=None):
        """
        Build Variational Problem

        This is called on each run of `run_model()` to reset the parameters
        of the simulation as necessary.
        """
        logger.debug("Constructing k_x field")
        params = self.lam_true if params is None else params
        proj_vals = self.reconstruct(params, log=True)
        kx = fem.Function(self.V)
        kx.x.array[:] = proj_vals

        logger.debug("Assembling bilinear form")
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        a = (
            u * v * ufl.dx
            + self.solve_ts * kx * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        self.bilinear_form = fem.form(a)
        A = fem.petsc.assemble_matrix(self.bilinear_form, bcs=[self.boundary_condition])
        A.assemble()

        logger.debug("Assembling linear form")
        if self.forcing_expression is None:
            self.f = fem.Constant(self.domain, 0.0)
            L = (self.u_n + self.solve_ts * self.f) * v * ufl.dx
        else:
            self.f = self.forcing_expression
            self.f.t = 0.0
            self.w = fem.Function(self.V)
            L = (self.u_n + self.solve_ts * self.w) * v * ufl.dx
            self.w.interpolate(self.f.eval)

        self.linear_form = fem.form(L)
        self.b = fem.petsc.create_vector(self.linear_form)

        # Petsc solver
        logger.debug("Initializing solver")
        solver = PETSc.KSP().create(self.domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        self.solver = solver

    def forward_model(
        self,
        x0: List[float],
        times: np.ndarray,
        lam: np.ndarray,
        fname=None,
        sample_ts=0.1,
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
        if fname is not None:
            xdmf = io.XDMFFile(self.domain.comm, fname, "w")
            xdmf.write_mesh(self.domain)
            snap_counter = 0.0

        # Set initial conditions (how to do?)
        self.u_n.x.array[:] = np.array(x0).ravel()
        self.uh.x.array[:] = np.array(x0).ravel()

        # Create variational problme to solve given thermal diff. field lam
        self._create_variational_problem(np.array(lam))

        sol = np.zeros((len(times), self.n_states))
        for i, t in enumerate(times):
            sol[i] = self.uh.x.array.copy().ravel()

            # Update forcing
            if self.forcing_expression is not None:
                self.f.t = t
                self.w.interpolate(self.f.eval)

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

            if fname is not None:
                if snap_counter >= sample_ts:
                    snap_counter += self.solve_ts
                    snap_counter = 0.0
                    if fname is not None:
                        xdmf.write_function(self.uh, self.t)

        if fname is not None:
            xdmf.close()

        return sol

    def take_snaps(self, data_df, sample_ts=0.01):
        """
        Take snapshots of data at a given time interval.
        Snapshots consist of data frame with columns t_{i} for
        each time step to be ploted, and each row being a specific
        index in the grid.
        """
        remainder = data_df["ts"] % sample_ts

        # Select rows where the remainder is close to zero (within a small tolerance)
        data = data_df[remainder < 1e-4]
        times = [f"t_{i}" for i, t in enumerate(data["ts"])]
        true_cols = [col for col in data.columns if col.startswith("q_lam_true")]
        data = data[true_cols].values.T

        return pd.DataFrame(data, columns=times)

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
        """
        
        """
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
            raise ImportError("Cannot create gif - pyvista not found")
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

    def k_x_mud_plot(self, iteration=0, figsize=(18, 5)):
        """
        Plot estimated and True k(x)
        """
        iteration = 0
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        self.plot_field(field=self.probs[iteration].mud_point, ax=ax[0])
        ax[0].set_title("$k^{MUD}(x)$")
        self.plot_field(field=self.lam_true, ax=ax[1])
        ax[1].set_title("$k^{\dagger}(x)$")
        ax[1].set_ylabel("")
        self.plot_field(
            field=self.probs[iteration].mud_point, diff=self.lam_true, ax=ax[2]
        )
        ax[2].set_title("Error")
        ax[2].set_ylabel("")
        fig.tight_layout
