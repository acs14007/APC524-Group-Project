from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import scipy.io


class Environment:
    poisson_iteration_steps = 200

    def __init__(
        self,
        dx=0.05,
        len_x=2.0,
        len_y=2.0,
        dt: float = 0.01,
        boundary_conditions=None,
        objects=None,
        F=(0.0, 0.0),
        rho=1.0,
        nu=0.1,
        dtype=None,
    ):
        self.dtype = np.longdouble if dtype is None else dtype

        self.dx = dx
        self.dy = dx

        self.nx = int(len_x / dx) + 1
        self.ny = int(len_y / dx) + 1

        self.x = np.linspace(0, len_x, self.nx, dtype=self.dtype)
        self.y = np.linspace(0, len_y, self.ny, dtype=self.dtype)

        self.dt = dt

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.u = np.zeros((self.ny, self.nx), dtype=self.dtype)
        self.un = np.zeros((self.ny, self.nx), dtype=self.dtype)

        self.v = np.zeros((self.ny, self.nx), dtype=self.dtype)
        self.vn = np.zeros((self.ny, self.nx), dtype=self.dtype)

        self.b = np.zeros((self.ny, self.nx), dtype=self.dtype)
        self.p = np.ones((self.ny, self.nx), dtype=self.dtype)
        self.pn = np.ones((self.ny, self.nx), dtype=self.dtype)

        self.F = F

        self.boundary_conditions = (
            boundary_conditions if boundary_conditions is not None else []
        )

        self.objects = objects if objects is not None else []

        self.rho = rho
        self.nu = nu

        self.u_list = []
        self.v_list = []
        self.p_list = []
        self.data_dict = None

        self.stepcount = 0

    def update_b_matrix(self):
        # Reset the b matrix
        self.b = self.b * 0

        self.b[1:-1, 1:-1] = self.rho * (
            1
            / self.dt
            * (
                (self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx)
                + (self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy)
            )
            - ((self.u[1:-1, 2:] - self.u[1:-1, 0:-2]) / (2 * self.dx)) ** 2
            - 2
            * (
                (self.u[2:, 1:-1] - self.u[0:-2, 1:-1])
                / (2 * self.dy)
                * (self.v[1:-1, 2:] - self.v[1:-1, 0:-2])
                / (2 * self.dx)
            )
            - ((self.v[2:, 1:-1] - self.v[0:-2, 1:-1]) / (2 * self.dy)) ** 2
        )
        return self

    def update_pressure_matrix(self):
        for q in range(self.poisson_iteration_steps):
            self.pn = self.p.copy()
            self.p[1:-1, 1:-1] = (
                (self.pn[1:-1, 2:] + self.pn[1:-1, 0:-2]) * self.dy**2
                + (self.pn[2:, 1:-1] + self.pn[0:-2, 1:-1]) * self.dx**2
            ) / (2 * (self.dx**2 + self.dy**2)) - self.dx**2 * self.dy**2 / (
                2 * (self.dx**2 + self.dy**2)
            ) * self.b[
                1:-1, 1:-1
            ]
        return self

    def u_matrix_update_step(self):
        self.u[1:-1, 1:-1] = (
            self.un[1:-1, 1:-1]
            - self.un[1:-1, 1:-1]
            * self.dt
            / self.dx
            * (self.un[1:-1, 1:-1] - self.un[1:-1, 0:-2])
            - self.vn[1:-1, 1:-1]
            * self.dt
            / self.dy
            * (self.un[1:-1, 1:-1] - self.un[0:-2, 1:-1])
            - self.dt
            / (2 * self.rho * self.dx)
            * (self.p[1:-1, 2:] - self.p[1:-1, 0:-2])
            + self.nu
            * (
                self.dt
                / self.dx**2
                * (self.un[1:-1, 2:] - 2 * self.un[1:-1, 1:-1] + self.un[1:-1, 0:-2])
                + self.dt
                / self.dy**2
                * (self.un[2:, 1:-1] - 2 * self.un[1:-1, 1:-1] + self.un[0:-2, 1:-1])
            )
            + self.F[0] * self.dt
        )

    def v_matrix_update_step(self):
        self.v[1:-1, 1:-1] = (
            self.vn[1:-1, 1:-1]
            - self.un[1:-1, 1:-1]
            * self.dt
            / self.dx
            * (self.vn[1:-1, 1:-1] - self.vn[1:-1, 0:-2])
            - self.vn[1:-1, 1:-1]
            * self.dt
            / self.dy
            * (self.vn[1:-1, 1:-1] - self.vn[0:-2, 1:-1])
            - self.dt
            / (2 * self.rho * self.dy)
            * (self.p[2:, 1:-1] - self.p[0:-2, 1:-1])
            + self.nu
            * (
                self.dt
                / self.dx**2
                * (self.vn[1:-1, 2:] - 2 * self.vn[1:-1, 1:-1] + self.vn[1:-1, 0:-2])
                + self.dt
                / self.dy**2
                * (self.vn[2:, 1:-1] - 2 * self.vn[1:-1, 1:-1] + self.vn[0:-2, 1:-1])
            )
            + self.F[1] * self.dt
        )

    def set_boundary_conditions(self):
        for boundary_condition in self.boundary_conditions:
            boundary_condition.apply_boundary_condition(self)

        return self

    def set_object_boundary_conditions(self):
        for object in self.objects:
            object.apply_boundary_conditions(self)

    def take_one_step(self, save = False):
        self.un = self.u.copy()
        self.vn = self.v.copy()

        self.u_matrix_update_step()
        self.v_matrix_update_step()

        self.set_boundary_conditions()
        self.set_object_boundary_conditions()

        self.update_b_matrix()
        self.update_pressure_matrix()
        self.stepcount += 1

        if save:
            self.u_list.append(self.u.copy())
            self.v_list.append(self.v.copy())
            self.p_list.append(self.p.copy())

        return self

    def run_many_steps(self, steps: int, save = False):
        if save:
            for _ in tqdm(range(steps), total=steps):
                self.take_one_step(save = True)
            u_total_array = np.stack(self.u_list, axis=-1)
            v_total_array = np.stack(self.v_list, axis=-1)
            p_total_array = np.stack(self.p_list, axis=-1)
            t_total_array = np.arange(0, steps * self.dt, self.dt)
            x_total_array = self.x.copy()
            y_total_array = self.y.copy()
            dict_keys = ['u_star', 'v_star', 'p_star', 't', 'x_star', 'y_star']
            dict_values = [u_total_array, v_total_array, p_total_array, 
                           t_total_array, x_total_array, y_total_array]
            self.data_dict = dict(zip(dict_keys, dict_values))
        else:
            for _ in tqdm(range(steps), total=steps):
                self.take_one_step()

    def save_model_run(self, filepath: str):
        if self.data_dict is None:
            raise Exception("No data to save. Run the model first with run_many_steps() with argument save = True")
        else:
            scipy.io.savemat(filepath, self.data_dict)

    def plot_quiver_plot(
        self, filepath=None, show_objects=True, number_of_items_to_skip=3, title=None
    ):
        width = 10
        height = width / self.nx * self.ny
        fig, ax = plt.subplots(figsize=(width, height), dpi=200)
        ax.contourf(self.X, self.Y, self.p, alpha=0.5, cmap="Pastel1")
        # ax.quiver(self.X, self.Y, self.u, self.v, color="xkcd:dark grey")
        ax.quiver(
            self.X[::number_of_items_to_skip, ::number_of_items_to_skip],
            self.Y[::number_of_items_to_skip, ::number_of_items_to_skip],
            self.u[::number_of_items_to_skip, ::number_of_items_to_skip],
            self.v[::number_of_items_to_skip, ::number_of_items_to_skip],
            color="xkcd:dark grey",
        )

        if show_objects:
            for object in self.objects:
                object.plot_object(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if title is None:
            ax.set_title("Velocity field")
        else:
            ax.set_title(title)
        ax.set_aspect("equal")

        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath)

        plt.show()

    def plot_streamline_plot(
        self,
        filepath=None,
        show_objects=True,
        return_figure=False,
        vmin=None,
        vmax=None,
        title=None,
    ):
        width = 10
        height = width / self.nx * self.ny
        fig, ax = plt.subplots(figsize=(width, height), dpi=200)
        if (vmin is not None) and (vmax is not None):
            ax.contourf(
                self.X, self.Y, self.p, alpha=0.5, cmap="Pastel2", vmin=vmin, vmax=vmax
            )
        else:
            ax.contourf(self.X, self.Y, self.p, alpha=0.5, cmap="Pastel2")
        ax.streamplot(
            self.X, self.Y, self.u, self.v, color="xkcd:purple", arrowstyle="->"
        )

        if show_objects:
            for object in self.objects:
                object.plot_object(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if title is None:
            ax.set_title("Velocity field")
        else:
            ax.set_title(title)
        ax.set_aspect("equal")

        plt.tight_layout()

        if filepath is not None:
            plt.savefig(filepath)

        if return_figure:
            plt.clf()
            return fig

        plt.show()
