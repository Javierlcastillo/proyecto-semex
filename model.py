import agentpy as ap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches
from route import Route
from car import Car

from defined_routes import routes
from defined_tlconnections import tlconnections

class Renderer(ap.Model):
    """
    Minimal model that keeps arrays of agents moving along assigned routes.
    """
    fig: Figure
    ax: Axes
    
    routes: list[Route]
    car_patches: list[patches.Rectangle] = []

    def setup(self):
        self.routes = routes[0:1]
        self.cars = [
            Car(
                self, 
                r, 
                [tlc for tlc in tlconnections if tlc.route == r]
            ) for r in self.routes
        ]

        self.fig, self.ax = plt.subplots(figsize=(8,8), squeeze=True) # type: ignore
        self.ax.set_aspect('equal')
        self.ax.set_xlim(100, 700)
        self.ax.set_ylim(0, 600)

        
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'cyan', 'magenta']
        for i, r in enumerate(self.routes):
            r.plot(self.ax, color=colors[i % len(colors)])
        
        self.fig.show()

    def _heading(self, car: Car) -> float:
        p0 = car.route.pos_at(car.s)
        p1 = car.route.pos_at(car.s + car.ds)
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    def plot(self):
        for car in self.cars:
            car.plot(self.ax)

        self.fig.canvas.draw_idle() # type: ignore
        self.fig.canvas.flush_events()

    def step(self):
        # Advance cars, then repaint them
        for car in self.cars:
            if hasattr(car, "step"):
                car.step()

        self.plot()
