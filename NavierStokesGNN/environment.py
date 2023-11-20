from abc import abstractmethod, ABC
from typing import List, Tuple

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


class BoundaryCondition:
    def __init__(self, boundary_type, location, value=None):
        self.boundary_type = boundary_type
        self.location = location
        self.value = value


class Object(ABC):
    @abstractmethod
    def get_mesh(self) -> List[Tuple[int, int]]:
        """
        Abstract method to get the mesh points of the object in the field that might block the flow.
        Should return a list of (x, y) tuples representing the mesh points of the object.
        """
        pass


class Rectangle(Object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_mesh(self) -> List[tuple]:
        """
        A list of (x, y) tuples representing the mesh points of the object.
        """
        mesh = []
        for i in range(self.width):
            for j in range(self.height):
                mesh.append((self.x + i, self.y + j))
        return mesh


class Environment:
    def __init__(self,
                 width_meters,
                 height_meters,
                 points_per_meter,
                 data_type=tf.float32,
                 boundary_conditions=None):
        self.number_of_width_cells = int(width_meters * points_per_meter)
        self.number_of_height_cells = int(height_meters * points_per_meter)

        # Initialize velocity and pressure fields
        self.u = tf.Variable(tf.zeros([self.number_of_height_cells, self.number_of_width_cells]), dtype=data_type)
        self.v = tf.Variable(tf.zeros([self.number_of_height_cells, self.number_of_width_cells]), dtype=data_type)
        self.pressure = tf.Variable(tf.zeros([self.number_of_height_cells, self.number_of_width_cells]),
                                    dtype=data_type)

        self.boundary_conditions = boundary_conditions
        if self.boundary_conditions is None:
            self.set_default_boundary_conditions()

        self.objects = []

    def set_default_boundary_conditions(self, default_velocity=1.0):
        """
        Set default boundary conditions for a pipe flow.
        """
        # Fixed velocity at the inlet (left boundary)
        velocity = default_velocity
        left_boundary = BoundaryCondition('fixed-velocity', 'left', velocity)

        # Fixed velocity at the outlet (right boundary)
        right_boundary = BoundaryCondition('fixed-velocity', 'right', velocity)

        # No-slip (zero velocity) condition at other boundaries
        top_boundary = BoundaryCondition('no-slip', 'top')
        bottom_boundary = BoundaryCondition('no-slip', 'bottom')

        self.boundary_conditions = [left_boundary, top_boundary, bottom_boundary, right_boundary]

    @tf.function
    def set_boundary_condition(self, boundary_condition: BoundaryCondition):
        """
        Apply a single boundary condition.
        """
        if boundary_condition.boundary_type == 'fixed-velocity':
            # For edges such as left and right edge
            # Only applies the boundary condition to the direction perpendicular to the edge
            if boundary_condition.location == 'left':
                self.u[:, 0].assign(tf.fill([self.number_of_height_cells], boundary_condition.value))
            elif boundary_condition.location == 'right':
                self.u[:, -1].assign(tf.fill([self.number_of_height_cells], boundary_condition.value))
            elif boundary_condition.location == 'top':
                self.v[0, :].assign(tf.fill([self.number_of_width_cells], boundary_condition.value))
            elif boundary_condition.location == 'bottom':
                self.v[-1, :].assign(tf.fill([self.number_of_width_cells], boundary_condition.value))

        elif boundary_condition.boundary_type == 'no-slip':
            if boundary_condition.location == 'top':
                self.u[0, :].assign(tf.zeros(self.number_of_width_cells))
                self.v[0, :].assign(tf.zeros(self.number_of_width_cells))
            elif boundary_condition.location == 'bottom':
                self.u[-1, :].assign(tf.zeros(self.number_of_width_cells))
                self.v[-1, :].assign(tf.zeros(self.number_of_width_cells))
            elif boundary_condition.location == 'right':
                self.u[:, -1].assign(tf.zeros(self.number_of_height_cells))
                self.v[:, -1].assign(tf.zeros(self.number_of_height_cells))
            elif boundary_condition.location == 'left':
                self.u[:, 0].assign(tf.zeros(self.number_of_height_cells))
                self.v[:, 0].assign(tf.zeros(self.number_of_height_cells))

        elif boundary_condition.boundary_type == 'perfect_slip':
            if boundary_condition.location == 'top':
                # Assign boundary_condition.value to the top
                self.u[0, :].assign(tf.fill([self.number_of_width_cells], boundary_condition.value))
                self.v[0, :].assign(tf.zeros(self.number_of_width_cells))
            elif boundary_condition.location == 'bottom':
                # Assign boundary_condition.value to the bottom
                self.u[-1, :].assign(tf.fill([self.number_of_width_cells], boundary_condition.value))
                self.v[-1, :].assign(tf.zeros(self.number_of_width_cells))
            elif boundary_condition.location == 'right':
                # Assign boundary_condition.value to the right
                self.u[:, -1].assign(tf.zeros(self.number_of_height_cells))
                self.v[:, -1].assign(tf.fill([self.number_of_height_cells], boundary_condition.value))
            elif boundary_condition.location == 'left':
                # Assign boundary_condition.value to the left
                self.u[:, 0].assign(tf.zeros(self.number_of_height_cells))
                self.v[:, 0].assign(tf.fill([self.number_of_height_cells], boundary_condition.value))

    @tf.function
    def set_all_boundary_conditions(self):
        """
        Apply all boundary conditions.
        """
        if self.boundary_conditions:
            for boundary_condition in self.boundary_conditions:
                self.set_boundary_condition(boundary_condition)

    def add_object(self, obj: Rectangle):
        """
        Add an object to the environment and apply boundary conditions.
        """
        self.objects.append(obj)
        mesh = obj.get_mesh()
        for point in mesh:
            x, y = point

            # Apply no-slip condition on the object's points
            # TODO: Implement this in a more efficient way
            # self.u[y, x].assign(0)
            # self.v[y, x].assign(0)

    def plot(self, show=True, path=None):
        """
        Plot the pressure field and overlay streamlines for the velocity field.
        """
        # Convert TensorFlow tensors to numpy arrays for plotting
        u_np = self.u.numpy()
        v_np = self.v.numpy()
        pressure_np = self.pressure.numpy()

        # Create a meshgrid for plotting
        y_, x_ = np.mgrid[0:self.number_of_height_cells, 0:self.number_of_width_cells]

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.streamplot(x_, y_, u_np, v_np, color='blue', density=2, arrowstyle='->')
        plt.imshow(pressure_np, extent=(0, self.number_of_width_cells, 0, self.number_of_height_cells),
                   origin='lower', cmap='hot', alpha=0.5)
        plt.colorbar(label='Pressure')

        # Plot object points
        for obj in self.objects:
            mesh = obj.get_mesh()
            for point in mesh:
                plt.scatter(*point, color='black')

        plt.title('Streamlines over Pressure Field')
        if path is not None:
            plt.savefig(path)

        if show:
            plt.show()

        return plt


if __name__ == '__main__':
    print('Starting Pipe Simulation Test...')
    env = Environment(10, 5, 10)
    env.set_default_boundary_conditions()

    # Add a rectangular object to the center of the domain
    center_x = env.number_of_width_cells // 2
    center_y = env.number_of_height_cells // 2
    rect_obj = Rectangle(center_x - 1, center_y - 1, 2, 2)  # Example rectangle
    env.add_object(rect_obj)

    env.plot(path='example_environment.png')
    print('Boundary conditions set and plot generated with object')
