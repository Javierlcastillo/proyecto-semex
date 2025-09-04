import agentpy as ap
from route import Route, Point
import numpy as np
from matplotlib import patches, transforms, axes, figure
from typing import Optional
from traffic_light import TrafficLight, TLConnection, TrafficLightState

class Car(ap.Agent):
    route: Route
    tlconnections: list[TLConnection]

    position: Point
    ds: float

    maxRateOfAceleration: float
    width: float
    height: float
    is_colliding: bool

    patch: Optional[patches.Rectangle]

    @property
    def nextTLC(self):
        nearest: Optional[TLConnection] = None

        for tlc in self.tlconnections:
            if tlc.route == self.route and self.s < tlc.s and (nearest is None or tlc.s < nearest.s):
                nearest = tlc
        return nearest

    def setup(self, route: Route, tlconnections: list[TLConnection]): # pyright: ignore[reportIncompatibleMethodOverride]
        self.route = route
        self.tlconnections = tlconnections

        self.s = 0.0
        self.position = self.route.pos_at(self.s)

        self.maxRateOfAceleration = np.random.uniform(0.8, 1.2)

        self.ds = 1  # Amount of S that changes in a step

        self.width = 20
        self.height = 10
        self.is_colliding = False

    def step(self):

        if (self.s == self.nextTLC.s and self.nextTLC.traffic_light.state == TrafficLightState.RED):
            # do nothing
            print("whatup")
        else:
            self.s += self.ds
            self.position = self.route.pos_at(self.s)

    def plot(self, ax: axes.Axes):
        x, y = self.position
        w, h = self.width, self.height

        if not hasattr(self, "patch") or self.patch is None:
            self.patch = patches.Rectangle(
                (x - w/2, y - h/2), w, h,
                facecolor='black', edgecolor='white', linewidth=1, zorder=2
            )
            ax.add_patch(self.patch)  # Add patch to Axes
        else:
            self.patch.set_xy((x - w/2, y - h/2))

        angle = self.heading()
        self.patch.set_transform(
            transforms.Affine2D().rotate_around(x, y, angle) + ax.transData
        )        

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
