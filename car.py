# car.py
import agentpy as ap
import numpy as np
from typing import Optional, Iterable, Any
from matplotlib import patches, transforms  # used by plot()

from route import Route
from traffic_light import TLConnection, TrafficLightState
try:
    from qlearner import QLearner as QLearner  # real class if available
except Exception:
    class _QLearner:  # stub used only for typing / fallback
        def choose_action(self, *args, **kwargs):
            return None

class Car(ap.Agent):
    """Route-following car with route-agnostic predictive collision avoidance."""

    # --- init ---
    def __init__(self, model, route, tlconnections, id, q_learner=None):
        super().__init__(model)
        self.route = route
        self.tlconnections = list(tlconnections or [])
        self.id = id
        self.q_learner = q_learner

        # --- kinematics ---
        self.s = 0.0
        self.ds = 4.0
        self.v_max = 6.0
        self.free_accel = 0.25
        self.restart_speed = 1.2

        # --- geometry (define FIRST) ---
        self.width = 20.0
        self.height = 10.0

        # --- spacing knobs (can depend on height) ---
        self.queue_gap = 12.0            # desired standstill gap
        self.lane_tol = self.height * 0.6  # same-lane lateral tolerance
        self.t_headway = 0.40            # extra gap that scales with speed

        # --- render/state ---
        self.position = self.route.pos_at(self.s)
        self.angle = 0.0
        self.state = "moving"
        self.is_colliding = False
        self.patch = None


    # ---------- geometry ----------
    def heading_at(self, s: float) -> float:
        L = float(getattr(self.route, "length", 0.0))
        if L <= 1e-9:
            return 0.0
        eps = max(1e-3, 0.001 * L)
        s0 = (s - 0.5 * eps) % L
        s1 = (s + 0.5 * eps) % L
        x0, y0 = self.route.pos_at(float(s0))
        x1, y1 = self.route.pos_at(float(s1))
        return float(np.arctan2(y1 - y0, x1 - x0))

    def heading(self) -> float:
        return self.heading_at(self.s)

    def corners_at(self, s: float) -> np.ndarray:
        x, y = self.route.pos_at(float(s))
        w, h = self.width, self.height
        a = self.heading_at(float(s))
        c, s_ = np.cos(a), np.sin(a)
        hw, hh = w / 2.0, h / 2.0
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
        R = np.array([[c, -s_], [s_, c]], dtype=float)
        return local @ R.T + np.array([x, y], dtype=float)

    def corners(self) -> np.ndarray:
        return self.corners_at(self.s)

    @staticmethod
    def _sat_overlap(A: np.ndarray, B: np.ndarray) -> bool:
        """Separating Axis Theorem for convex polygons (rectangles here)."""
        def axes_from(poly: np.ndarray):
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                edge = p2 - p1
                n = np.array([-edge[1], edge[0]], dtype=float)
                ln = np.linalg.norm(n)
                if ln > 1e-12:
                    yield n / ln

        for poly in (A, B):
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                edge = p2 - p1
                axis = np.array([-edge[1], edge[0]], dtype=float)
                n = np.linalg.norm(axis)
                if n <= 1e-12:
                    continue
                axis /= n
                a0, a1 = (A @ axis).min(), (A @ axis).max()
                b0, b1 = (B @ axis).min(), (B @ axis).max()
                if a1 < b0 or b1 < a0:
                    return False
        return True

    # ---------- perception / avoidance ----------
    def perceive_neighbors(self, cars: Iterable["Car"], radio: float = 60.0) -> list[tuple[float, "Car"]]:
        out: list[tuple[float, Car]] = []
        px, py = self.position
        for o in cars:
            if o is self:
                continue
            qx, qy = o.position
            d = float(np.hypot(qx - px, qy - py))
            if d <= radio:
                out.append((d, o))
        out.sort(key=lambda t: t[0])
        return out

    def _is_in_front(self, other: "Car", fov_deg: float = 140.0) -> bool:
        """Whether 'other' is roughly ahead (within forward FOV)."""
        a = self.heading()
        dirx, diry = np.cos(a), np.sin(a)
        px, py = self.position
        ox, oy = other.position
        vx, vy = ox - px, oy - py
        d = np.hypot(vx, vy)
        if d <= 1e-6:
            return True  # touching -> treat as in front
        ux, uy = vx / d, vy / d
        return (ux * dirx + uy * diry) >= np.cos(np.deg2rad(fov_deg * 0.5))

    def overlap_now(self, other: "Car") -> bool:
        return self._sat_overlap(self.corners(), other.corners())

    def resolve_overlap_forward(self, cars: Iterable["Car"], max_push: float = 2.0, step: float = 0.2) -> bool:
        """Nudge forward along the route until we no longer overlap; commit if success."""
        L = float(getattr(self.route, "length", 0.0))
        if L <= 1e-9:
            return False
        s_test = float(self.s)
        moved = 0.0
        while moved < max_push:
            A = self.corners_at(s_test)
            if all(not self._sat_overlap(A, other.corners()) for other in cars if other is not self):
                self.s = s_test
                self.position = self.route.pos_at(self.s)
                self.angle = self.heading_at(self.s)
                return True
            s_test = (s_test + step) % L
            moved += step
        return False

    def will_collide_next(self, other: "Car", *, self_ds: Optional[float] = None) -> bool:
        """Predict overlap next tick (route-agnostic)."""
        if self is other:
            return False
        Ls = float(getattr(self.route, "length", 0.0))
        Lo = float(getattr(other.route, "length", 0.0))
        ds_self = self.ds if self_ds is None else float(self_ds)
        s1 = (self.s + ds_self) % Ls if Ls > 0 else self.s
        so1 = (other.s + other.ds) % Lo if Lo > 0 else other.s
        return self._sat_overlap(self.corners_at(s1), other.corners_at(so1))

    def hazard_ahead_next(self, other: "Car", *, preview_self: Optional[float] = None) -> bool:
        """Block only for hazards ahead, but still block if overlapping now."""
        if not self._is_in_front(other):
            return self.overlap_now(other)  # rear car: ignore unless overlapping now
        return self.will_collide_next(other, self_ds=preview_self)

    # ---------- lights ----------
    @property
    def nextTLC(self) -> Optional[TLConnection]:
        nearest: Optional[TLConnection] = None
        for tlc in self.tlconnections:
            if tlc.route is self.route and self.s < tlc.s:
                if nearest is None or tlc.s < nearest.s:
                    nearest = tlc
        return nearest

    # ---------- Q helper ----------
    def _choose_action_safe(self, state: Any):
        """Call _QLearner.choose_action with whatever signature it supports."""
        choose = getattr(self.q_learner, "choose_action", None)
        if not callable(choose):
            return None
        try:
            return choose(state, exploration=False)     # preferred
        except TypeError:
            try:
                return choose(state, False)             # positional flag
            except TypeError:
                try:
                    return choose(state)                # state only
                except TypeError:
                    return None
    def _long_lat(self, other):
        a = self.heading()
        dirx, diry = np.cos(a), np.sin(a)
        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]
        longi = dx*dirx + dy*diry                  # forward distance (center-to-center)
        lat   = -dx*diry + dy*dirx                 # lateral offset (right = +)
        return float(longi), float(lat)

    def too_close_ahead(self, other, preview_self=None) -> bool:
        # only same-lane-ish and ahead
        longi, lat = self._long_lat(other)
        if longi <= 0 or abs(lat) > self.lane_tol:
            return False
        # clearance between bumpers along forward axis
        ds_self = self.ds if preview_self is None else float(preview_self)
        clearance = longi - (self.width*0.5 + other.width*0.5)
        desired   = self.queue_gap + self.t_headway*ds_self
        return clearance < desired

    # ---------- main step ----------
    def step(self):
        L = float(getattr(self.route, "length", 0.0))
        if L <= 1e-9:
            return

        # (optional) Q-learner
        if self.q_learner:
            try:
                state = self.model.get_state(self) if hasattr(self.model, "get_state") else None
                action = self._choose_action_safe(state)
                if action is not None:
                    self.apply_q_action(action)
            except Exception:
                pass

        # traffic light stopping
        tlc = self.nextTLC
        if tlc is not None:
            ahead = (tlc.s - self.s) % L
            if ahead <= 14.0 and tlc.traffic_light.state in (TrafficLightState.RED, TrafficLightState.YELLOW):
                self.state = "stop"

        # route-agnostic predictive braking (only hazards AHEAD)
        if hasattr(self, "model") and hasattr(self.model, "cars"):
            for _, other in self.perceive_neighbors(self.model.cars, radio=60.0):
                if not self._is_in_front(other):
                    continue
                if self.will_collide_next(other) or self.too_close_ahead(other):
                    self.ds = max(0.0, self.ds - 0.6)
                    if self.will_collide_next(other) or self.too_close_ahead(other):
                        self.state = "stop"
                        self.position = self.route.pos_at(self.s)
                        self.angle = self.heading_at(self.s)
                        self.update_collision(self.model.cars)
                        return

        # --- resume if clear (after a red) ---
        if self.state == "stop":
            can_move = True

            # 1) still red/yellow? stay stopped
            if tlc is not None:
                ahead = (tlc.s - self.s) % L
                if ahead <= 14.0 and tlc.traffic_light.state in (TrafficLightState.RED, TrafficLightState.YELLOW):
                    can_move = False

            # 2) optional: clear nose-to-tail overlap with a tiny nudge
            if can_move and hasattr(self, "model") and hasattr(self.model, "cars"):
                cars = self.model.cars
                if any(self.overlap_now(o) for o in cars if o is not self):
                    self.resolve_overlap_forward(cars, max_push=2.0, step=0.2)

            # 3) >>> STEP 4 goes HERE <<<
            if can_move and hasattr(self, "model") and hasattr(self.model, "cars"):
                preview_speed = max(self.ds, self.restart_speed)
                for _, other in self.perceive_neighbors(self.model.cars, radio=60.0):
                    if not self._is_in_front(other):
                        continue  # rear/adjacent cars don't block us
                    if self.hazard_ahead_next(other, preview_self=preview_speed) or \
                    self.too_close_ahead(other, preview_self=preview_speed):
                        can_move = False
                        break

            # 4) finalize decision
            if not can_move:
                self.position = self.route.pos_at(self.s)
                self.angle = self.heading_at(self.s)
                self.update_collision(self.model.cars)
                return

            # cleared -> start rolling
            self.state = "moving"
            if self.ds < self.restart_speed:
                self.ds = self.restart_speed


        # gentle free-flow acceleration when clear
        if self.state == "moving" and self.ds < self.v_max:
            self.ds = min(self.v_max, self.ds + self.free_accel)

        # advance along route
        self.s = (self.s + self.ds) % L
        self.position = self.route.pos_at(self.s)
        self.angle = self.heading_at(self.s)

        # telemetry
        if hasattr(self, "model") and hasattr(self.model, "cars"):
            self.update_collision(self.model.cars)

    # ---------- misc ----------
    def update_collision(self, cars: Iterable["Car"]) -> None:
        A = self.corners()
        self.is_colliding = any(self._sat_overlap(A, o.corners()) for o in cars if o is not self)

    def plot(self, ax):
        """Draw car rectangle, always above routes."""
        x, y = self.position
        w, h = self.width, self.height
        TOP = 1000  # cars above everything

        if self.patch is None:
            self.patch = patches.Rectangle(
                (x - w / 2, y - h / 2), w, h,
                linewidth=1.0,
                edgecolor="none",
                facecolor="black",
                alpha=0.85,
                zorder=TOP,
            )
            ax.add_patch(self.patch)
        else:
            self.patch.set_xy((x - w / 2, y - h / 2))

        angle = self.heading()
        self.patch.set_transform(
            transforms.Affine2D().rotate_around(x, y, angle) + ax.transData
        )
        self.patch.set_zorder(TOP)

    def export_state(self):
        return {"id": self.id, "pos": self.position, "ds": self.ds, "angle": self.angle}

    def apply_q_action(self, action: Any):
        if action == "frenar":
            self.ds = max(0.0, self.ds - 0.5)
        elif action == "acelerar":
            self.ds += 0.5
        # "mantener" => no change

    # reward helper used elsewhere
    def compute_reward(self, car: "Car"):
        if car.is_colliding:
            return -100
        if car.ds == 0:
            return -5
        return +1
