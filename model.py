# model.py
import agentpy as ap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches

from route import Route
from car import Car
from defined_routes import routes, route_colors, route_widths
from defined_tlconnections import tlconnections

from network_manager import NetManager

class Renderer(ap.Model):
    """
    Renders routes, traffic lights and cars.
    - Routes are plotted here (we don't call Route.plot).
    - Colors/widths come from defined_routes (exported from Unity).
    """

    fig: Figure
    ax: Axes

    routes: list[Route]
    car_patches: list[patches.Rectangle] = []

    net: NetManager

    def setup(self):
        self.net = NetManager()
        self.net.start()

        self.routes = routes
        self.tlconnections = tlconnections  # Store tlconnections for Unity bridge
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

        # Plot routes directly
        default_palette = [
            "#FFD200", "#FF3B30", "#34C759", "#5856D6",
            "#A2845E", "#00BCD4", "#FF2D55", "#1E90FF"
        ]

        for i, r in enumerate(self.routes):
            c = route_colors[i] if i < len(route_colors) else default_palette[i % len(default_palette)]
            lw = route_widths[i] if i < len(route_widths) else 2.0

            pts = _get_polyline(r)
            if not pts:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self.ax.plot(xs, ys, color=c, linewidth=lw)

        # Traffic lights
        for tlc in tlconnections:
            if hasattr(tlc, "traffic_light") and hasattr(tlc.traffic_light, "plot"):
                tlc.traffic_light.plot(self.ax)

        self.fig.show()

    def plot(self):
        # Draw cars at current positions
        for car in self.cars:
            if hasattr(car, "plot"):
                car.plot(self.ax)

        self.fig.canvas.draw_idle()      # type: ignore
        self.fig.canvas.flush_events()   # type: ignore

    def step(self):
        # Advance cars, then repaint
        for car in self.cars:
            if hasattr(car, "step"):
                car.step()
        self.plot()
        self.push_state()

    def push_state(self):
        state = {
            "cars": [
                {
                    "id": id(car),
                    "position": car.route.pos_at(car.s),
                    "heading": self._heading(car),
                    "route_id": id(car.route)
                } for car in self.cars
            ],
        }

        self.net.push_state(state)
