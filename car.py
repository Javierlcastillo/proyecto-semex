import agentpy as ap
from route import Route, Point
import numpy as np

class Car(ap.Agent):
    route: Route
    position: Point
    ds: float

    maxRateOfAceleration: float
    width: float
    height: float
    is_colliding: bool

    def setup(self, route: Route): # pyright: ignore[reportIncompatibleMethodOverride]
        self.route = route

        self.s = 0.0
        self.position = self.route.pos_at(self.s)

        self.maxRateOfAceleration = np.random.uniform(0.8, 1.2)

        self.ds = 1  # Amount of S that changes in a step

        self.width = 10
        self.height = 20
        self.is_colliding = False

    def step(self):
        self.s += self.ds
        self.position = self.route.pos_at(self.s)

    def accelerate(self, rate: float):
        if np.abs(rate) < np.abs(self.maxRateOfAceleration):
            self.ds *= rate

    # --- Collision helpers on the agent ---

    def heading(self) -> float:
        p0 = self.route.pos_at(self.s)
        p1 = self.route.pos_at(self.s + self.ds)
        return float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))  # radians

    def corners(self) -> np.ndarray:
        # Oriented rectangle corners (counter-clockwise) in world coords
        x, y = self.position
        w, h = self.width, self.height
        a = self.heading()
        c, s = np.cos(a), np.sin(a)
        hw, hh = w/2.0, h/2.0
        local = np.array([
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh],
        ])
        R = np.array([[c, -s],[s, c]])
        return local @ R.T + np.array([x, y])

    @staticmethod
    def _sat_overlap(A: np.ndarray, B: np.ndarray) -> bool:
        # A and B: (4,2) arrays of rectangle corners
        def axes_from(poly: np.ndarray):
            edges = np.roll(poly, -1, axis=0) - poly
            axes = []
            for e in edges[:2]:  # two unique edge directions for a rectangle
                n = np.array([-e[1], e[0]])
                ln = np.linalg.norm(n)
                if ln > 1e-9:
                    axes.append(n / ln)
            return axes

        for axis in axes_from(A) + axes_from(B):
            projA = A @ axis
            projB = B @ axis
            if projA.max() < projB.min() or projB.max() < projA.min():
                return False
        return True

    def collides_with(self, other: "Car") -> bool:
        if self is other:
            return False
        return self._sat_overlap(self.corners(), other.corners())

    def update_collision(self, cars: list["Car"]) -> None:
        self.is_colliding = any(self.collides_with(o) for o in cars if o is not self)
