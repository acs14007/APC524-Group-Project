from abc import ABC, abstractmethod
from typing import Iterable

from NavierStokesGNN.boundary_condition import BoundaryCondition
import numpy as np


class Object(ABC):
    @abstractmethod
    def apply_boundary_conditions(self, environment):
        pass

    @abstractmethod
    def plot_object(self):
        pass

    @staticmethod
    def convert_x_coordinate_to_indices(x_coordinate, environment):
        """
        Convert the rectangle's physical coordinates to grid indices.
        """
        x_index = int(x_coordinate / environment.dx)
        return x_index

    @staticmethod
    def convert_y_coordinate_to_indices(y_coordinate, environment):
        """
        Convert the rectangle's physical coordinates to grid indices.
        """
        y_index = int(y_coordinate / environment.dy)
        return y_index

    @staticmethod
    def convert_coordinate_point_to_indices(point, environment):
        """
        Convert the rectangle's physical coordinates to grid indices.
        """
        x_index = Object.convert_y_coordinate_to_indices(point[0], environment.dx)
        y_index = Object.convert_y_coordinate_to_indices(point[1], environment.dy)
        return x_index, y_index


class Rectangle(Object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def apply_boundary_conditions(self, environment):
        """
        Applies a no-slip boundary condition to the top edge of the rectangle.
        """

        i1 = self.convert_x_coordinate_to_indices(self.x1, environment)
        i2 = self.convert_x_coordinate_to_indices(self.x2, environment)
        j1 = self.convert_y_coordinate_to_indices(self.y1, environment)
        j2 = self.convert_y_coordinate_to_indices(self.y2, environment)

        # Top Edge No Slip
        environment.u[j2, i1:i2] = 0
        environment.v[j2, i1:i2] = 0
        environment.p[j1, i1:i2] = environment.p[j1 + 1, i1:i2]

        # Bottom Edge No Slip
        environment.u[j1, i1:i2] = 0
        environment.v[j1, i1:i2] = 0
        environment.p[j1, i1:i2] = environment.p[j1 - 1, i1:i2]

        # Left Edge No Slip
        environment.u[j1:j2, i1] = 0
        environment.v[j1:j2, i1] = 0
        environment.p[j1:j2, i1] = environment.p[j1:j2, i1 - 1]

        # Right Edge No Slip
        environment.u[j1:j2, i2] = 0
        environment.v[j1:j2, i2] = 0
        environment.p[j1:j2, i2] = environment.p[j1:j2, i2 + 1]

        environment.u[j1:j2, i1:i2] = 0
        environment.v[j1:j2, i1:i2] = 0
        environment.p[j1 + 1 : j2 - 1, i1 + 1 : i2 - 1] = 0

    def plot_object(self, ax):
        ax.fill_between(
            [self.x1, self.x2], self.y1, self.y2, color="xkcd:dark grey", zorder=10
        )
