import agentpy as ap
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches
from route import Route
from car import Car

from defined_routes import routes
from defined_tlconnections import tlconnections

from network_manager import NetManager

class Renderer(ap.Model):        
    """
    Minimal model that keeps arrays of agents moving along assigned routes.
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
        self.cars = []
        self.spawn_step = {id(route): -10 for route in self.routes}  # Track last spawn step for each route
        self.current_step = 0

        # Debug: print initial positions of all cars
        print('Initial car positions:')
        for idx, car in enumerate(self.cars):
            print(f'Car {idx}: {car.position}')

            

        self.fig, self.ax = plt.subplots(figsize=(8,8), squeeze=True) # type: ignore
        self.ax.set_aspect('equal')
        self.ax.set_xlim(100, 700)
        self.ax.set_ylim(0, 600)

        
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'cyan', 'magenta']
        for i, route in enumerate(self.routes):
            route.plot(self.ax, color=colors[i % len(colors)])

        for tlc in tlconnections:
            tlc.traffic_light.plot(self.ax)
        
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

        # Respawn cars for all routes if spawn position is clear (no car too close ahead), total cars < nCars, and spawn interval passed
        for route in self.routes:
            route_id = id(route)
            spawn_pos = route.pos_at(0.0)
            cars_ahead = [car for car in self.cars if car.route == route and car.s < car.width]
            if (
                not cars_ahead
                and len(self.cars) < self.p.nCars
                and self.current_step - self.spawn_step[route_id] >= 200
            ):
                self.cars.append(
                    Car(
                        self,
                        route,
                        [tlc for tlc in self.tlconnections if tlc.route == route]
                    )
                )
                self.spawn_step[route_id] = self.current_step

        self.current_step += 1

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
