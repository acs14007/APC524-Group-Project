import sys
from navier_stokes_fdm import Environment
from navier_stokes_fdm import Circle
import navier_stokes_fdm.boundary_condition as bc

U = 1  # m/s
rho_air = 1.225 / 2  # kg/m³
nu_air = 3e-5  # m²/s

boundary_conditions = [
    # bc.TopSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    # bc.BottomSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    bc.LeftSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    # bc.RightSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    bc.TopSideFreeSlipBoundaryCondition(),
    bc.RightSideFreeSlipBoundaryCondition(),
    bc.BottomSideFreeSlipBoundaryCondition(),
]

dimension = 0.005
objects = [Circle(0.0125, 0.02, dimension)]

a = Environment(
    F=(1.0, 0.0),
    len_x=0.06,
    len_y=0.04,
    dt=0.00000015,
    dx=0.0001,
    boundary_conditions=boundary_conditions,
    objects=objects,
    rho=rho_air,
    nu=nu_air,
)

for i in range(14):
    a.run_many_steps(30)
    a.plot_streamline_plot(
        title="",
        filepath=f"../Figures/cylinder_example_timesteps/streamline{i:02d}.png",
    )

    if i == 8:
        a.plot_streamline_plot(
            title="", filepath="../Figures/cylinder_example_streamline.png"
        )
        a.plot_quiver_plot(title="", filepath="../Figures/cylinder_example_quiver.png")
