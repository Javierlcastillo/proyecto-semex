import agentpy as ap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches
from route import Route
from car import Car
from traffic_light import TrafficLight, TLConnection

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

        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(100, 700)
        self.ax.set_ylim(0, 600)

        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'cyan', 'magenta']

        for i, r in enumerate(self.routes):
            pts = r.sample_even(400)
            self.ax.plot(pts[:,0], pts[:,1], color=colors[i % len(colors)], linewidth=2, zorder=1)
        
        self.fig.show()

    def _heading(self, car: Car) -> float:
        p0 = car.route.pos_at(car.s)
        p1 = car.route.pos_at(car.s + car.ds)
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    def plot(self):
        for car in self.cars:
            car.plot(self.ax)

        for c in self.traffic_light_connections:
            c.traffic_light.plot(self.ax)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def get_traffic_lights_in_route(self, r: Route) -> list[TLConnection]:
        return [tlc for tlc in self.traffic_light_connections if tlc.route == r]

    def step(self):
        # Advance cars, then repaint them
        for car in self.cars:
            if hasattr(car, "step"):
                car.step()

        self.plot()
