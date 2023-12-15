import numpy as np
from NavierStokesGNN.environment import Environment
from NavierStokesGNN.boundary_condition import TopSideNoSlipBoundaryCondition

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
    assert np.array_equal(env.p[-1, :], env.p[-2, :]), "Top row of p matrix should be equal to second-to-top row"

# Run this test
test_top_side_no_slip_boundary_condition()


