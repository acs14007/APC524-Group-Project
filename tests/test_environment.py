import pytest
import numpy as np
from NavierStokesGNN.environment import Environment
import copy

def test_environment_initialization_defaults():
    """
    Test if the Environment class initializes with the correct default values.
    This verifies that without any custom parameters, the class attributes are set to their expected default states.
    """
    env = Environment()
    assert env.dx == 0.05, "Default dx should be 0.05"
    assert env.dy == 0.05, "Default dy should be 0.05 (same as dx)"
    assert env.nx == int(2.0 / 0.05) + 1, "Default nx should be computed based on len_x and dx"
    assert env.ny == int(2.0 / 0.05) + 1, "Default ny should be computed based on len_y and dx"
    assert np.array_equal(env.x, np.linspace(0, 2.0, env.nx, dtype=env.dtype)), "Default x array should be a linspace from 0 to len_x"
    assert np.array_equal(env.y, np.linspace(0, 2.0, env.ny, dtype=env.dtype)), "Default y array should be a linspace from 0 to len_y"
    assert env.dt == 0.01, "Default dt should be 0.01"
    assert np.all(env.u == 0), "Default u matrix should be initialized to zeros"
    assert np.all(env.v == 0), "Default v matrix should be initialized to zeros"
    assert np.all(env.b == 0), "Default b matrix should be initialized to zeros"
    assert np.all(env.p == 1), "Default p matrix should be initialized to ones"
    assert env.F == (0.0, 0.0), "Default F should be (0.0, 0.0)"
    assert env.boundary_conditions == [], "Default boundary_conditions should be an empty list"
    assert env.objects == [], "Default objects should be an empty list"
    assert env.rho == 1.0, "Default rho should be 1.0"
    assert env.nu == 0.1, "Default nu should be 0.1"
    assert env.stepcount == 0, "Default stepcount should be 0"
    assert env.dtype == np.longdouble, "Default dtype should be np.longdouble"

def test_update_b_matrix():
    env = Environment()
    # Modify u and v matrices in a way that should change b matrix
    env.u[1:-1, 1:-1] = np.random.rand(env.ny - 2, env.nx - 2)
    env.v[1:-1, 1:-1] = np.random.rand(env.ny - 2, env.nx - 2)
    initial_b = env.b.copy()
    env.update_b_matrix()
    assert not np.array_equal(initial_b, env.b), "B matrix should be updated"

def test_update_pressure_matrix():
    """
    Test the update_pressure_matrix method of the Environment class.
    Ensures that the pressure matrix 'p' is updated correctly.
    """
    env = Environment()
    env.b += 1  # Set up a non-zero state for the b matrix
    initial_p = env.p.copy()
    env.update_pressure_matrix()
    assert not np.array_equal(initial_p, env.p), "Pressure matrix should be updated"

def test_take_one_step():
    env = Environment()
    # Modify initial conditions to ensure change after step
    env.u[1:-1, 1:-1] = 1
    env.v[1:-1, 1:-1] = 1
    initial_state = copy.deepcopy(env)
    env.take_one_step()
    assert not np.array_equal(initial_state.u, env.u), "U matrix should be updated"
    assert not np.array_equal(initial_state.v, env.v), "V matrix should be updated"

