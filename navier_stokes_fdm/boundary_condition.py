from abc import ABC, abstractmethod


class BoundaryCondition(ABC):
    @abstractmethod
    def apply_boundary_condition(self, environment):
        pass


class PeriodicBoundaryCondition(BoundaryCondition):
    @abstractmethod
    def apply_b_boundary_condition(self, environment):
        pass

    @abstractmethod
    def apply_p_boundary_condition(self, environment):
        pass

    @abstractmethod
    def apply_u_boundary_condition(self, environment):
        pass

    @abstractmethod
    def apply_v_boundary_condition(self, environment):
        pass

    def apply_boundary_condition(self, environment):
        self.apply_b_boundary_condition(environment)
        self.apply_p_boundary_condition(environment)
        self.apply_u_boundary_condition(environment)
        self.apply_v_boundary_condition(environment)


class FixedVelocityBoundaryCondition(BoundaryCondition, ABC):
    def __init__(self, u_value, v_value):
        self.u_value = u_value
        self.v_value = v_value


class RightSideFixedVelocityBoundaryCondition(FixedVelocityBoundaryCondition, ABC):
    def apply_boundary_condition(self, environment):
        environment.u[1:-1, -1] = self.u_value
        environment.v[1:-1, -1] = self.v_value


class LeftSideFixedVelocityBoundaryCondition(FixedVelocityBoundaryCondition):
    def apply_boundary_condition(self, environment):
        environment.u[1:-1, 0] = self.u_value
        environment.v[1:-1, 0] = self.v_value


class TopSideFixedVelocityBoundaryCondition(FixedVelocityBoundaryCondition):
    def apply_boundary_condition(self, environment):
        environment.u[-1, 1:-1] = self.u_value
        environment.v[-1, 1:-1] = self.v_value


class BottomSideFixedVelocityBoundaryCondition(FixedVelocityBoundaryCondition):
    def apply_boundary_condition(self, environment):
        environment.u[0, 1:-1] = self.u_value
        environment.v[0, 1:-1] = self.v_value


class TopSideNoSlipBoundaryCondition(BoundaryCondition):
    def apply_boundary_condition(self, environment):
        environment.u[-1, :] = 0
        environment.v[-1, :] = 0
        environment.p[-1, :] = environment.p[-2, :]


class BottomSideNoSlipBoundaryCondition(BoundaryCondition):
    def apply_boundary_condition(self, environment):
        environment.u[0, :] = 0
        environment.v[0, :] = 0
        environment.p[0, :] = environment.p[1, :]


class TopSideFreeSlipBoundaryCondition(BoundaryCondition):
    def apply_boundary_condition(self, environment):
        environment.v[-1, :] = 0
        environment.p[-1, :] = environment.p[-2, :]


class BottomSideFreeSlipBoundaryCondition(BoundaryCondition):
    def apply_boundary_condition(self, environment):
        environment.v[0, :] = 0
        environment.p[0, :] = environment.p[1, :]


class RightSideFreeSlipBoundaryCondition(BoundaryCondition):
    def apply_boundary_condition(self, environment):
        environment.u[:, -1] = 0
        environment.p[:, -1] = environment.p[:, -2]


class LeftSidePeriodicBoundaryCondition(PeriodicBoundaryCondition):
    def apply_b_boundary_condition(self, environment):
        environment.b[1:-1, 0] = environment.rho * (
            1
            / environment.dt
            * (
                (environment.u[1:-1, 1] - environment.u[1:-1, -1])
                / (2 * environment.dx)
                + (environment.v[2:, 0] - environment.v[0:-2, 0]) / (2 * environment.dy)
            )
            - (
                (environment.u[1:-1, 1] - environment.u[1:-1, -1])
                / (2 * environment.dx)
            )
            ** 2
            - 2
            * (
                (environment.u[2:, 0] - environment.u[0:-2, 0])
                / (2 * environment.dy)
                * (environment.v[1:-1, 1] - environment.v[1:-1, -1])
                / (2 * environment.dx)
            )
            - ((environment.v[2:, 0] - environment.v[0:-2, 0]) / (2 * environment.dy))
            ** 2
        )

    def apply_p_boundary_condition(self, environment):
        environment.p[1:-1, 0] = (
            (environment.pn[1:-1, 1] + environment.pn[1:-1, -1]) * environment.dy**2
            + (environment.pn[2:, 0] + environment.pn[0:-2, 0]) * environment.dx**2
        ) / (
            2 * (environment.dx**2 + environment.dy**2)
        ) - environment.dx**2 * environment.dy**2 / (
            2 * (environment.dx**2 + environment.dy**2)
        ) * environment.b[
            1:-1, 0
        ]

    def apply_u_boundary_condition(self, environment):
        environment.u[1:-1, 0] = (
            environment.un[1:-1, 0]
            - environment.un[1:-1, 0]
            * environment.dt
            / environment.dx
            * (environment.un[1:-1, 0] - environment.un[1:-1, -1])
            - environment.vn[1:-1, 0]
            * environment.dt
            / environment.dy
            * (environment.un[1:-1, 0] - environment.un[0:-2, 0])
            - environment.dt
            / (2 * environment.rho * environment.dx)
            * (environment.p[1:-1, 1] - environment.p[1:-1, -1])
            + environment.nu
            * (
                environment.dt
                / environment.dx**2
                * (
                    environment.un[1:-1, 1]
                    - 2 * environment.un[1:-1, 0]
                    + environment.un[1:-1, -1]
                )
                + environment.dt
                / environment.dy**2
                * (
                    environment.un[2:, 0]
                    - 2 * environment.un[1:-1, 0]
                    + environment.un[0:-2, 0]
                )
            )
            + environment.F[0] * environment.dt
        )

    def apply_v_boundary_condition(self, environment):
        environment.v[1:-1, 0] = (
            environment.vn[1:-1, 0]
            - environment.un[1:-1, 0]
            * environment.dt
            / environment.dx
            * (environment.vn[1:-1, 0] - environment.vn[1:-1, -1])
            - environment.vn[1:-1, 0]
            * environment.dt
            / environment.dy
            * (environment.vn[1:-1, 0] - environment.vn[0:-2, 0])
            - environment.dt
            / (2 * environment.rho * environment.dy)
            * (environment.p[2:, 0] - environment.p[0:-2, 0])
            + environment.nu
            * (
                environment.dt
                / environment.dx**2
                * (
                    environment.vn[1:-1, 1]
                    - 2 * environment.vn[1:-1, 0]
                    + environment.vn[1:-1, -1]
                )
                + environment.dt
                / environment.dy**2
                * (
                    environment.vn[2:, 0]
                    - 2 * environment.vn[1:-1, 0]
                    + environment.vn[0:-2, 0]
                )
            )
            + environment.F[1] * environment.dt
        )


class RightSidePeriodicBoundaryCondition(PeriodicBoundaryCondition):
    def apply_b_boundary_condition(self, environment):
        environment.b[1:-1, -1] = environment.rho * (
            1
            / environment.dt
            * (
                (environment.u[1:-1, 0] - environment.u[1:-1, -2])
                / (2 * environment.dx)
                + (environment.v[2:, -1] - environment.v[0:-2, -1])
                / (2 * environment.dy)
            )
            - (
                (environment.u[1:-1, 0] - environment.u[1:-1, -2])
                / (2 * environment.dx)
            )
            ** 2
            - 2
            * (
                (environment.u[2:, -1] - environment.u[0:-2, -1])
                / (2 * environment.dy)
                * (environment.v[1:-1, 0] - environment.v[1:-1, -2])
                / (2 * environment.dx)
            )
            - ((environment.v[2:, -1] - environment.v[0:-2, -1]) / (2 * environment.dy))
            ** 2
        )

    def apply_p_boundary_condition(self, environment):
        environment.p[1:-1, -1] = (
            (environment.pn[1:-1, 0] + environment.pn[1:-1, -2]) * environment.dy**2
            + (environment.pn[2:, -1] + environment.pn[0:-2, -1]) * environment.dx**2
        ) / (
            2 * (environment.dx**2 + environment.dy**2)
        ) - environment.dx**2 * environment.dy**2 / (
            2 * (environment.dx**2 + environment.dy**2)
        ) * environment.b[
            1:-1, -1
        ]

    def apply_u_boundary_condition(self, environment):
        environment.u[1:-1, -1] = (
            environment.un[1:-1, -1]
            - environment.un[1:-1, -1]
            * environment.dt
            / environment.dx
            * (environment.un[1:-1, -1] - environment.un[1:-1, -2])
            - environment.vn[1:-1, -1]
            * environment.dt
            / environment.dy
            * (environment.un[1:-1, -1] - environment.un[0:-2, -1])
            - environment.dt
            / (2 * environment.rho * environment.dx)
            * (environment.p[1:-1, 0] - environment.p[1:-1, -2])
            + environment.nu
            * (
                environment.dt
                / environment.dx**2
                * (
                    environment.un[1:-1, 0]
                    - 2 * environment.un[1:-1, -1]
                    + environment.un[1:-1, -2]
                )
                + environment.dt
                / environment.dy**2
                * (
                    environment.un[2:, -1]
                    - 2 * environment.un[1:-1, -1]
                    + environment.un[0:-2, -1]
                )
            )
            + environment.F[0] * environment.dt
        )

    def apply_v_boundary_condition(self, environment):
        environment.v[1:-1, -1] = (
            environment.vn[1:-1, -1]
            - environment.un[1:-1, -1]
            * environment.dt
            / environment.dx
            * (environment.vn[1:-1, -1] - environment.vn[1:-1, -2])
            - environment.vn[1:-1, -1]
            * environment.dt
            / environment.dy
            * (environment.vn[1:-1, -1] - environment.vn[0:-2, -1])
            - environment.dt
            / (2 * environment.rho * environment.dy)
            * (environment.p[2:, -1] - environment.p[0:-2, -1])
            + environment.nu
            * (
                environment.dt
                / environment.dx**2
                * (
                    environment.vn[1:-1, 0]
                    - 2 * environment.vn[1:-1, -1]
                    + environment.vn[1:-1, -2]
                )
                + environment.dt
                / environment.dy**2
                * (
                    environment.vn[2:, -1]
                    - 2 * environment.vn[1:-1, -1]
                    + environment.vn[0:-2, -1]
                )
            )
            + environment.F[1] * environment.dt
        )
