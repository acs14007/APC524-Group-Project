"""
snakeviz simulation_profile.prof # Visualise the results of the efficieny. 

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

