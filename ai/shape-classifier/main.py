from collections import namedtuple
from dataclasses import dataclass, field
import sys
import math
from typing import Literal
import torch
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from timeit import default_timer as timer


def print_versions():
    dependencies = {"torch": torch, "numpy": np}
    pythonVersion = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}-{sys.version_info.releaselevel}"
    )
    print("Dependencies")
    print(f"Using Python: Version {pythonVersion} from {sys.prefix}")
    for name, dependency in dependencies.items():
        print(f"{name}: {dependency.__version__}")


Point = namedtuple("Point", "x y")


@dataclass
class RectBoundaries:
    min: Point
    max: Point


@dataclass
class Rectangle:
    width: float
    height: float
    offset: npt.NDArray
    boundaries: RectBoundaries = field(init=False)

    def __post_init__(self):
        rect_size = 0.5 * np.array([self.width, self.height])
        self.boundaries = RectBoundaries(min=Point(*(-rect_size + self.offset)), max=Point(*(rect_size + self.offset)))


class CircleCloud:
    def __init__(self, vertices_amount: int, radius: float, offset: npt.NDArray) -> None:
        if vertices_amount <= 4:
            raise ValueError("The vertices amount has to be equal or greater than four.")
        if offset.shape != (2,):
            raise ValueError(f"Cannot use an offset of dimension {offset.shape}")
        self.vertices_amount = vertices_amount
        self.radius = radius
        self.offset = offset
        degrees_rad = np.linspace(0, 2 * math.pi, vertices_amount, endpoint=False)
        self.vertices: npt.NDArray = radius * np.array((np.sin(degrees_rad), np.cos(degrees_rad))).T + offset

    def _lerp(self, min: float, max: float, scale: float) -> float:
        return min - scale + max * scale

    def _clamp_explosion(self, rect: Rectangle, factor: float, max_d: float) -> npt.NDArray:
        exploded_points = (self.vertices - self.offset) * self._lerp(1, max_d, factor) + self.offset  # type: ignore
        return np.array(
            [
                (
                    min(rect.boundaries.max.x, max(p[0], rect.boundaries.min.x)),
                    min(rect.boundaries.max.y, max(p[1], rect.boundaries.min.y)),
                )
                for p in exploded_points
            ]
        )

    def _ray_cast_explosion(self, rect: Rectangle, factor: float, max_d: float) -> npt.NDArray:
        point_scale = [1.0] * self.vertices.shape[0]

        scale = self._lerp(1, max_d, factor)
        for idx, p in enumerate(self.vertices):
            a_x_min = (rect.boundaries.min.x - circle.offset[0]) / (p[0] - circle.offset[0])
            a_x_max = (rect.boundaries.max.x - circle.offset[0]) / (p[0] - circle.offset[0])
            a_y_min = (rect.boundaries.min.y - circle.offset[1]) / (p[1] - circle.offset[1])
            a_y_max = (rect.boundaries.max.y - circle.offset[1]) / (p[1] - circle.offset[1])

            min_a = min(
                [alpha for alpha in [a_x_min, a_x_max, a_y_min, a_y_max] if not (alpha < 0 or math.isinf(alpha))]
            )

            point_scale[idx] = min(min_a, scale)

        return (self.vertices - self.offset) * np.array(point_scale)[:, np.newaxis] + self.offset  # type: ignore

    def explode(
        self, rect: Rectangle, factor: float, explosion_type: Literal["clamp", "ray_cast"] = "ray_cast"
    ) -> npt.NDArray:
        rect_size = 0.5 * np.array([rect.width, rect.height])
        abs_origin = np.absolute(self.offset - rect.offset)  # type: ignore
        circle_inside_rect_bounds = np.all((abs_origin + self.radius) <= rect_size)
        if not circle_inside_rect_bounds:
            print("Circle is not contained inside the rectangle!")
        else:
            print("Circle is inside the rectangle")

        max_d = float(np.linalg.norm(abs_origin + rect_size, 2))
        explode_functions = {"clamp": self._clamp_explosion, "ray_cast": self._ray_cast_explosion}
        print(f"{max_d=}")
        return explode_functions[explosion_type](rect=rect, factor=factor, max_d=max_d)


if __name__ == "__main__":
    print_versions()

    vertices_amount = 48
    radius = 1

    rect = Rectangle(8 * radius, 4 * radius, np.array([1.0, 0.0]))
    circle = CircleCloud(vertices_amount, radius, np.array([-1, -1]))
    start = timer()
    rect_verts = circle.explode(rect, 0.9, "clamp")
    end = timer()
    print(f"Time to compute ray casts is {end - start}s")
    fig, ax = plt.subplots()

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw major and minor grid lines
    ax.grid(which="both", color="grey", linewidth=1, linestyle="-", alpha=0.2)
    ax.axis("equal")

    # Draw arrows
    arrow_fmt = dict(markersize=4, color="black", clip_on=False)
    ax.plot((1), (0), marker=">", transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker="^", transform=ax.get_xaxis_transform(), **arrow_fmt)
    ax.scatter(circle.vertices[:, 0], circle.vertices[:, 1])
    ax.scatter(rect_verts[:, 0], rect_verts[:, 1])
    ax.add_patch(
        patches.Rectangle(
            rect.offset - np.array([rect.width / 2, rect.height / 2]),  # type: ignore
            rect.width,
            rect.height,
            alpha=0.5,
            facecolor="none",
            fill=False,
        )
    )
    ax.add_patch(patches.Circle(circle.offset, radius=circle.radius, alpha=0.5, facecolor="none", fill=False))
    plt.show()
