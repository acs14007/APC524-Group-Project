import numpy as np
from navier_stokes_fdm import Environment
from navier_stokes_fdm import TopSideNoSlipBoundaryCondition
from navier_stokes_fdm import RightSideFixedVelocityBoundaryCondition
from navier_stokes_fdm import LeftSideFixedVelocityBoundaryCondition
from navier_stokes_fdm import TopSideFixedVelocityBoundaryCondition
from navier_stokes_fdm import BottomSideFixedVelocityBoundaryCondition
from navier_stokes_fdm import BottomSideNoSlipBoundaryCondition


def test_right_side_fixed_velocity_boundary_condition():
    env = Environment()
    u_value, v_value = 4, 4
    boundary_condition = RightSideFixedVelocityBoundaryCondition(u_value, v_value)

    boundary_condition.apply_boundary_condition(env)

    # Assert that the right side of u and v matrices are set to the specified values
    assert np.all(
        env.u[1:-1, -1] == u_value
    ), "Right side of u matrix should be set to u_value"
    assert np.all(
        env.v[1:-1, -1] == v_value
    ), "Right side of v matrix should be set to v_value"


def test_left_side_fixed_velocity_boundary_condition():
    env = Environment()
    u_value, v_value = 1, 1
    boundary_condition = LeftSideFixedVelocityBoundaryCondition(u_value, v_value)

    boundary_condition.apply_boundary_condition(env)

    assert np.all(
        env.u[1:-1, 0] == u_value
    ), "Left side of u matrix should be set to u_value"
    assert np.all(
        env.v[1:-1, 0] == v_value
    ), "Left side of v matrix should be set to v_value"


def test_top_side_fixed_velocity_boundary_condition():
    env = Environment()
    u_value, v_value = 2, 2
    boundary_condition = TopSideFixedVelocityBoundaryCondition(u_value, v_value)

    boundary_condition.apply_boundary_condition(env)

    assert np.all(
        env.u[-1, 1:-1] == u_value
    ), "Top row of u matrix should be set to u_value"
    assert np.all(
        env.v[-1, 1:-1] == v_value
    ), "Top row of v matrix should be set to v_value"


def test_bottom_side_fixed_velocity_boundary_condition():
    env = Environment()
    u_value, v_value = 3, 3
    boundary_condition = BottomSideFixedVelocityBoundaryCondition(u_value, v_value)

    boundary_condition.apply_boundary_condition(env)

    assert np.all(
        env.u[0, 1:-1] == u_value
    ), "Bottom row of u matrix should be set to u_value"
    assert np.all(
        env.v[0, 1:-1] == v_value
    ), "Bottom row of v matrix should be set to v_value"


def test_bottom_side_no_slip_boundary_condition():
    env = Environment()
    boundary_condition = BottomSideNoSlipBoundaryCondition()

    boundary_condition.apply_boundary_condition(env)

    assert np.all(env.u[0, :] == 0), "Bottom row of u matrix should be 0"
    assert np.all(env.v[0, :] == 0), "Bottom row of v matrix should be 0"
    assert np.array_equal(
        env.p[0, :], env.p[1, :]
    ), "Bottom row of p matrix should be equal to second-to-bottom row"


def test_top_side_no_slip_boundary_condition():
    """
    Test the TopSideNoSlipBoundaryCondition class.
    This checks if applying this boundary condition sets the top row of u, v, and p matrices to the correct values.
    """
    env = Environment()
    boundary_condition = TopSideNoSlipBoundaryCondition()

    # Apply the boundary condition
    boundary_condition.apply_boundary_condition(env)

    # Check if the top row of u and v matrices is set to 0
    assert np.all(env.u[-1, :] == 0), "Top row of u matrix should be 0"
    assert np.all(env.v[-1, :] == 0), "Top row of v matrix should be 0"

    # Check if the top row of p matrix is copied from the second-to-top row
    assert np.array_equal(
        env.p[-1, :], env.p[-2, :]
    ), "Top row of p matrix should be equal to second-to-top row"
