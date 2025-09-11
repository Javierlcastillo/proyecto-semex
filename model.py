import agentpy as ap
import numpy as np
from typing import Dict, Tuple, List
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches
from route import Route
from car import Car

from defined_routes import routes
from defined_tlconnections import tlconnections

from network_manager import NetManager

class Model(ap.Model):
    """
    Minimal model that keeps arrays of agents moving along assigned routes.
    """
    fig: Figure
    ax: Axes
    
    routes: list[Route]
    car_patches: list[patches.Rectangle] = []

    _step_idx: int = 0
    render_every: int = 10

    net: NetManager

    collision_flags: Dict[Car, bool] = {}

    def setup(self):
        self.net = NetManager()
        self.net.start()

        self.routes = routes
        self.tlconnections = tlconnections

        # Mode switches (with defaults)
        p = getattr(self, "p", {})
        self.training_enabled: bool = bool(p.get('train', True))
        render = bool(p.get('render', True))
        self.render_every: int = int(p.get('render_every', (20 if render else 0)))
        self.policy_dir: str = str(p.get('policy_dir', 'checkpoints'))
        # Only autosave during training
        self.autosave_interval: int = int(p.get('autosave_interval', (300 if self.training_enabled else 0)))

        # Create cars
        self.cars = [
            Car(
                self,
                route=r,
                tlconnections=[tlc for tlc in tlconnections if tlc.route == r]
            ) for r in self.routes
        ]

        # Set up plotting only if rendering
        self.fig = None
        self.ax = None
        if self.render_every > 0:
            self.fig, self.ax = plt.subplots(figsize=(8, 8), squeeze=True)  # type: ignore
            self.ax.set_aspect('equal')  # type: ignore
            self.ax.set_xlim(100, 700)   # type: ignore
            self.ax.set_ylim(0, 600)     # type: ignore
            plt.ion()

        self._step_idx = 0

    def _heading(self, car: Car) -> float:
        p0 = car.route.pos_at(car.s)
        p1 = car.route.pos_at(car.s + car.ds)
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    def plot(self):
        if not self.ax:
            return
        for car in self.cars:
            car.plot(self.ax)  # type: ignore[arg-type]
        self.fig.canvas.draw_idle()      # type: ignore[union-attr]
        self.fig.canvas.flush_events()   # type: ignore[union-attr]

    def _compute_collision_flags(self) -> None:
        """Broadphase (grid) + SAT narrowphase once per step for all cars."""
        cars = self.cars
        n = len(cars)
        if n == 0:
            self.collision_flags = {}
            return

        # Precompute corners and AABBs
        corners: List[np.ndarray] = [c._corners for c in cars]
        aabbs: List[Tuple[float, float, float, float]] = []
        for C in corners:
            xs = C[:, 0]; ys = C[:, 1]
            aabbs.append((float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())))

        # Uniform grid hashing on AABBs
        cell = max(max(c.width, c.height) for c in cars) * 2.0
        if cell <= 0:
            cell = 10.0
        grid: Dict[Tuple[int, int], List[int]] = {}
        for i, (xmin, xmax, ymin, ymax) in enumerate(aabbs):
            gx0, gx1 = int(np.floor(xmin / cell)), int(np.floor(xmax / cell))
            gy0, gy1 = int(np.floor(ymin / cell)), int(np.floor(ymax / cell))
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    grid.setdefault((gx, gy), []).append(i)

        # Candidate pairs from shared cells
        pairs: set[Tuple[int, int]] = set()
        for ids in grid.values():
            k = len(ids)
            if k < 2:
                continue
            for a in range(k):
                ia = ids[a]
                for b in range(a + 1, k):
                    ib = ids[b]
                    # Quick AABB overlap test to prune
                    ax0, ax1, ay0, ay1 = aabbs[ia]
                    bx0, bx1, by0, by1 = aabbs[ib]
                    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                        continue
                    pairs.add((min(ia, ib), max(ia, ib)))

        # Narrowphase SAT on candidates
        collided = [False] * n
        for ia, ib in pairs:
            if cars[ia]._sat_overlap(corners[ia], corners[ib]):
                collided[ia] = True
                collided[ib] = True

        # Publish per-car flags
        self.collision_flags = {cars[i]: collided[i] for i in range(n)}

    def step(self):
        # 1) Decision + movement
        for car in self.cars:
            car.step()

        # 2) Shared caches (collisions, TTC) if you have them
        if hasattr(self, "_compute_collision_flags"):
            self._compute_collision_flags()  # type: ignore
        if hasattr(self, "_compute_ttc_cache"):
            self._compute_ttc_cache()        # type: ignore

        # 3) Render sparsely
        self._step_idx += 1
        if self.render_every > 0 and (self._step_idx % self.render_every == 0):
            self.plot()

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
