from NavierStokesGNN import Environment
import NavierStokesGNN.boundary_condition as bc

boundary_conditions = [bc.TopSideNoSlipBoundaryCondition(),
                       bc.BottomSideNoSlipBoundaryCondition(),
                       bc.LeftSidePeriodicBoundaryCondition(),
                       bc.RightSidePeriodicBoundaryCondition()]

a = Environment(F=(1.0, 0.0), dt=0.001, boundary_conditions=boundary_conditions)

a.run_many_steps(5000)

a.plot_quiver_plot(filepath="../Figures/pipe_flow_example.png",
                   number_of_items_to_skip=3,
                   title="")
# a.plot_streamline_plot()
