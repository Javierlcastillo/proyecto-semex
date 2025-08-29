import agentpy as ap
from route import Route, Point
from typing import Any
import numpy as np

class Car(ap.Agent):
    route: Route
    position: Point
    ds: float

    maxRateOfAceleration: float
    width: float
    height: float

    def setup(self, route: Route): # pyright: ignore[reportIncompatibleMethodOverride]
        self.route = route

        self.s = 0.0
        self.position = self.route.pos_at(self.s)

        self.maxRateOfAceleration = np.random.uniform(0.8, 1.2)

        self.ds = 1 # Amount of S that changes in a step

        self.width = 10
        self.height = 20

    def step(self):
        self.s += self.ds
        self.position = self.route.pos_at(self.s)

    def accelerate(self, rate: float):
        if np.abs(rate) < np.abs(self.maxRateOfAceleration):
            self.ds *= rate
        