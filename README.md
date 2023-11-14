# APC524-Group-Project

Cylinder wake flow is an incredibly relevant problem in computational fluid dynamics that demonstrates key phenomena such as boundary layer separation and vortex shedding of fluid flowing around a blunt object. This problem can be approached by solving the **Navier-Stokes (NS) equation**. 

In this project we propose to (1) develop an implementation of a two-dimensional NS solver to simulate the cylinder wake flow and (2) train a physics-informed neural network to move forward in time using the first time-steps of the simulation.

This project contains the following key pieces:

* An implementation of a NS solver to simulate the two-dimensional cylinder wake flow.
* Train and implement a PINN approximating a solution of the two-dimensional cylinder wake flow.
* A comparison of the speed and accuracy of each method.
* Implementation of version control to document individual contribution, track project progress, and organize source code versions.
* Unit tests for each key function as well as detailed documentation for each file.
* Continuous testing using GitHub Actions.