# APC524: Implementing a Navier-Stokes Solver and Physics Informed Neural
Network for Simulating Two-Dimensional Fluid Flow Around a Cylinder

* An implementation of a modular Navier-Stokes solver using the Finite Difference Method.
* A Physics Informed Neural Network that approximates a Navier-Stokes solver.
*  Unit tests and automated testing 

## Final Report and Slides
### [Final Report](project_reports/final_report/final_report.pdf)
### [Slides](project_reports/final_report/final_slides.pdf)

## Example Simulation Result
![Example Plot](Figures/cylinder_example_timesteps/streamline05.png)

## Example Simulation Code
```python
from navier_stokes_fdm import Environment
from navier_stokes_fdm import Rectangle
import navier_stokes_fdm.boundary_condition as bc


U = 1  # m/s
dimension = 0.005

boundary_conditions = [
    bc.TopSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    bc.BottomSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    bc.LeftSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
    bc.RightSideFixedVelocityBoundaryCondition(u_value=U, v_value=0),
]


x1, y1 = 0.0125, (0.04 / 2) - (dimension / 2)
objects = [Rectangle(x1, y1, x1 + dimension, y1 + dimension)]


a = Environment(
    F=(1.0, 0.0),
    len_x=0.06,
    len_y=0.04,
    dt=0.00000015,
    dx=0.0001,
    boundary_conditions=boundary_conditions,
    objects=objects,
    rho=0.6125  # kg/m³
    nu=3e-5  # m²/s
)

a.run_many_steps(480)
a.plot_streamline_plot(title="", filepath="../Figures/box_example_streamline.png")

```
