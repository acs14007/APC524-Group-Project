from navier_stokes_fdm import Environment
import navier_stokes_fdm.boundary_condition as bc

boundary_conditions = [
    bc.TopSideNoSlipBoundaryCondition(),
    bc.BottomSideNoSlipBoundaryCondition(),
    bc.LeftSidePeriodicBoundaryCondition(),
    bc.RightSidePeriodicBoundaryCondition(),
]

a = Environment(F=(1.0, 0.0), dt=0.001, boundary_conditions=boundary_conditions)

a.run_many_steps(5000)

a.plot_quiver_plot(title=None, filepath="../Figures/pipe_flow_example_quiver.png")
a.plot_streamline_plot(
    title=None, filepath="../Figures/pipe_flow_example_streamline.png"
)
