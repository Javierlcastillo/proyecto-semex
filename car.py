import agentpy as ap
from route import Route, Point
from typing import Any

class Car(ap.Agent):
    route: Route
    position: Point
    
    def setup(self, route: Route): # pyright: ignore[reportIncompatibleMethodOverride]
        self.route = route

        self.s = 0.0
        self.position = self.route.pos_at(self.s)
        

        self.speed = 1.0 # units per step

    def step(self):
        self.s += self.speed
        self.position = self.route.pos_at(self.s)