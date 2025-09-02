import agentpy as ap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches, transforms
from route import Route
from car import Car

from defined_routes import routes


class Model(ap.Model):
    """
    Minimal model that keeps arrays of agents moving along assigned routes.
    """
    fig: Figure
    ax: Axes
    
    routes: list[Route]
    car_patches: list[patches.Rectangle] = []

    def setup(self):
        self.routes = routes
        self.cars = [Car(self, r) for r in self.routes]

        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(100, 700)
        self.ax.set_ylim(0, 600)

        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'cyan', 'magenta']

        for i, r in enumerate(self.routes):
            pts = r.sample_even(400)
            self.ax.plot(pts[:,0], pts[:,1], color=colors[i % len(colors)], linewidth=2, zorder=1)

        for car in self.cars:
            x, y = car.position
            w, h = car.width, car.height
            angle = self._heading(car)

            rect = patches.Rectangle(
                (x - w/2, y - h/2), h, w,
                facecolor='black', edgecolor='white', linewidth=1, zorder=2
            )
            rect.set_transform(
                transforms.Affine2D().rotate_around(x, y, angle) + self.ax.transData
            )
            self.ax.add_patch(rect)
            self.car_patches.append(rect)

        plt.ion()
        self.fig.show()

    def _heading(self, car: Car) -> float:
        p0 = car.route.pos_at(car.s)
        p1 = car.route.pos_at(car.s + car.ds)
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    def plot(self):
        for car, rect in zip(self.cars, self.car_patches):
            x, y = car.position
            w, h = car.width, car.height
            rect.set_xy((x - w/2, y - h/2))
            angle = self._heading(car)
            rect.set_transform(
                transforms.Affine2D().rotate_around(x, y, angle) + self.ax.transData
            )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def step(self):
        # Advance cars, then repaint them
        for car in self.cars:
            if hasattr(car, "step"):
                car.step()
        self.plot()
