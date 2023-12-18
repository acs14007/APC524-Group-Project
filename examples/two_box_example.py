from navier_stokes_fdm import Environment
from navier_stokes_fdm import Rectangle
import navier_stokes_fdm.boundary_condition as bc


U = 1  # m/s
rho_air = 1.225 / 2  # kg/m³
nu_air = 3e-5  # m²/s


boundary_conditions = [
    bc.TopSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    bc.BottomSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    bc.LeftSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    bc.RightSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
]

dimension = 0.005
x1, y1 = 0.0125, (0.04 / 2) - (dimension / 2)

objects = [Rectangle(x1, y1 - 0.01, x1 + dimension, y1 - 0.01 + dimension),
           Rectangle(x1, y1 + 0.01, x1 + dimension, y1 + 0.01 + dimension)]

L = objects[0].y2 - objects[0].y1
Reynolds = (rho_air * U * L) / nu_air
print(f"Reynolds = {Reynolds}")

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

a.run_many_steps(240)
a.plot_streamline_plot(filepath="../Figures/two_box_example_streamline.png")
a.plot_quiver_plot(filepath="../Figures/two_box_example_quiver.png")