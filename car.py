import agentpy as ap
import numpy as np
from matplotlib import patches, transforms, axes
from typing import Optional, Any

from route import Route, Point
from traffic_light import TLConnection, TrafficLightState


class Car(ap.Agent):
    route: Route
    tlconnections: list[TLConnection]

    position: Point
    s: float
    ds: float

    maxRateOfAceleration: float
    width: float
    height: float
    is_colliding: bool

    patch: Optional[patches.Rectangle]

    @property
    def nextTLC(self) -> Optional[TLConnection]:
        """Closest upcoming TLConnection along this car's route."""
        nearest: Optional[TLConnection] = None
        for tlc in self.tlconnections:
            if tlc.route is self.route and self.s < tlc.s and (nearest is None or tlc.s < nearest.s):
                nearest = tlc
        return nearest

    def setup(self, route: Route, tlconnections: list[TLConnection]):  # pyright: ignore[reportIncompatibleMethodOverride]
        self.route = route
        self.tlconnections = tlconnections

        # movement/physics
        self.ds = 1.0
        self.maxRateOfAceleration = float(np.random.uniform(0.8, 1.2))

        # geometry/state
        self.width = 20.0
        self.height = 10.0
        self.is_colliding = False

        # spawn
        self.s = 0.0
        pos0 = getattr(self.route, "json_start", None)
        self.position = (float(pos0[0]), float(pos0[1])) if pos0 is not None else self.route.pos_at(self.s)

    def step(self):
        L = float(getattr(self.route, "length", 0.0))
        if L <= 1e-9:
            return

        # --- stop at red/yellow lights, handling loops correctly ---
        STOP_DIST = 12.0     # how far before the stop line we brake
        SAFE_HEAD = 2.0      # small grace zone around the line
        WINDOW    = STOP_DIST + SAFE_HEAD

        for c in self.tlconnections:
            # forward distance along the route from car.s to the stop line
            ahead = (c.s - self.s) % L          # in [0, L)
            if ahead <= WINDOW:
                if c.traffic_light.state in (TrafficLightState.RED, TrafficLightState.YELLOW):
                    # hold position this tick
                    self.position = self.route.pos_at(self.s)
                    return

        # --- normal advance ---
        self.s += self.ds
        if self.s >= L:
            self.s -= L
        self.position = self.route.pos_at(self.s)


    def plot(self, ax: axes.Axes):
        x, y = self.position
        w, h = self.width, self.height

        if not hasattr(self, "patch") or self.patch is None:
            self.patch = patches.Rectangle(
                (x - w/2, y - h/2), w, h,
                facecolor="black", edgecolor="white", linewidth=1, zorder=2
            )
            ax.add_patch(self.patch)
        else:
            self.patch.set_xy((x - w/2, y - h/2))

        angle = self.heading()
        self.patch.set_transform(transforms.Affine2D().rotate_around(x, y, angle) + ax.transData)

    def accelerate(self, rate: float):
        if abs(rate) < abs(self.maxRateOfAceleration):
            self.ds *= rate

    def heading(self) -> float:
        L = float(getattr(self.route, "length", 0.0))
        if L <= 1e-9:
            return 0.0
        eps = max(1e-3, 0.001 * L)
        s0 = max(0.0, min(self.s - 0.5 * eps, L))
        s1 = max(0.0, min(self.s + 0.5 * eps, L))
        if s1 == s0:
            s0 = max(0.0, self.s - eps)
            s1 = min(L,   self.s + eps)
        x0, y0 = self.route.pos_at(s0)
        x1, y1 = self.route.pos_at(s1)
        return float(np.arctan2(y1 - y0, x1 - x0))

<<<<<<< Updated upstream
    def corners(self) -> np.ndarray[Any, Any]:
        x, y = self.position
        w, h = self.width, self.height
        a = self.heading()
        c, s = np.cos(a), np.sin(a)
        hw, hh = w / 2.0, h / 2.0
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + np.array([x, y])

    @staticmethod
    def _sat_overlap(A: np.ndarray[Any, Any], B: np.ndarray[Any, Any]) -> bool:
        def axes_from(poly: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
            edges = np.roll(poly, -1, axis=0) - poly
            axes: list[np.ndarray[Any, Any]] = []
            for e in edges[:2]:
                n = np.array([-e[1], e[0]])
                ln = np.linalg.norm(n)
                if ln > 1e-9:
                    axes.append(n / ln)  # type: ignore
            return axes  # type: ignore

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
=======
    # reward helper used elsewhere
    def compute_reward(self, car: "Car"):
        if car.is_colliding:
            return -100
        if car.ds == 0:
            return -5
        return +1
>>>>>>> Stashed changes
