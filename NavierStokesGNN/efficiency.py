"""
Fluid Flow Simulation Script

This Python script performs fluid flow simulations using the Navier-Stokes equations. It sets up an environment with defined parameters, boundary conditions, and objects. The core simulation logic is encapsulated in the `run_simulation` function.

Key Components:
- Imports necessary modules and classes.
- `run_simulation` initializes the environment, applies boundary conditions, introduces objects, and runs the simulation.
- The main section profiles the simulation's performance using cProfile and saves data to 'simulation_profile.prof'.

This script is part of a broader fluid dynamics simulation project, facilitating simulations, profiling, and optimization.
"""


import cProfile
from NavierStokesGNN.environment import Environment
from NavierStokesGNN.boundary_condition import (
    TopSideNoSlipBoundaryCondition,
    BottomSideNoSlipBoundaryCondition,
    LeftSideFixedVelocityBoundaryCondition,
    RightSideFixedVelocityBoundaryCondition,
    TopSideFixedVelocityBoundaryCondition,
    BottomSideFixedVelocityBoundaryCondition,
    LeftSidePeriodicBoundaryCondition,
    RightSidePeriodicBoundaryCondition
)
from NavierStokesGNN.object import Rectangle

def run_simulation():
    # Initialize the environment
    env = Environment(dx=0.05, dt=0.01)

    # Add boundary conditions
    env.boundary_conditions.append(TopSideNoSlipBoundaryCondition())
    env.boundary_conditions.append(BottomSideNoSlipBoundaryCondition())
    env.boundary_conditions.append(LeftSideFixedVelocityBoundaryCondition(u_value=1.0, v_value=0.0))
    env.boundary_conditions.append(RightSideFixedVelocityBoundaryCondition(u_value=1.0, v_value=0.0))
    env.boundary_conditions.append(TopSideFixedVelocityBoundaryCondition(u_value=0.0, v_value=1.0))
    env.boundary_conditions.append(BottomSideFixedVelocityBoundaryCondition(u_value=0.0, v_value=1.0))
    env.boundary_conditions.append(LeftSidePeriodicBoundaryCondition())
    env.boundary_conditions.append(RightSidePeriodicBoundaryCondition())

    # Add objects
    env.objects.append(Rectangle(0.5, 0.5, 1.5, 1.5))

    # Run the simulation for a specified number of steps
    env.run_many_steps(100)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation()
    profiler.disable()
    profiler.dump_stats('simulation_profile.prof')

