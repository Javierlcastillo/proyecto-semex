import agentpy as ap
from route import Route, Point
import numpy as np
from matplotlib import patches, transforms, axes
from typing import Optional, Any
from traffic_light import TLConnection

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

    # car.py
    def setup(self, route: Route, tlconnections: list[TLConnection]):  # pyright: ignore[reportIncompatibleMethodOverride]
        self.route = route
        self.tlconnections = tlconnections

        # --- movement/physics params (needed by step) ---
        self.ds = 1.0                               # <-- REQUIRED by step()
        self.maxRateOfAceleration = np.random.uniform(0.8, 1.2)

        # --- car geometry / state used by plot/collision ---
        self.width = 20.0
        self.height = 10.0
        self.is_colliding = False

        # --- spawn position (prefer first coord from routes.json) ---
        pos0 = getattr(self.route, "json_start", None)   # Optional[Tuple[float, float]]
        self.s = 0.0
        if pos0 is not None:
            self.position = (float(pos0[0]), float(pos0[1]))
        else:
            self.position = self.route.pos_at(self.s)


    def step(self):
        L = getattr(self.route, "length", 0.0)
        if L <= 1e-9: 
            return
        self.s += self.ds
        if self.s >= L:               # wrap (or set self.s = 0.0 to respawn)
            self.s -= L
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
        L = getattr(self.route, "length", 0.0)
        if L <= 1e-9:
            return 0.0
        eps = max(1e-3, 0.001 * L)
        s0 = max(0.0, min(self.s - 0.5*eps, L))
        s1 = max(0.0, min(self.s + 0.5*eps, L))
        if s1 == s0:
            s0 = max(0.0, self.s - eps)
            s1 = min(L,   self.s + eps)
        x0, y0 = self.route.pos_at(s0)
        x1, y1 = self.route.pos_at(s1)
        return float(np.arctan2(y1 - y0, x1 - x0))


    def corners(self) -> np.ndarray[Any, Any]:
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
    def _sat_overlap(A: np.ndarray[Any, Any], B: np.ndarray[Any, Any]) -> bool:
        # A and B: (4,2) arrays of rectangle corners
        def axes_from(poly: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
            edges = np.roll(poly, -1, axis=0) - poly
            axes = []
            for e in edges[:2]:  # two unique edge directions for a rectangle
                n = np.array([-e[1], e[0]])
                ln = np.linalg.norm(n)
                if ln > 1e-9:
                    axes.append(n / ln) # type: ignore
            return axes # type: ignore

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
