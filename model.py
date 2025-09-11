# model.py
import agentpy as ap
import numpy as np
from typing import Tuple, Sequence, Any, cast
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from route import Route
from car import Car
from defined_routes import routes, route_colors, route_widths, route_names
from defined_tlconnections import tlconnections, traffic_lights
# (TrafficLightState import not needed here)


# ---------- helpers ----------
def _route_len(r) -> float:
    return float(getattr(r, "length", 0.0) or 0.0)

def _spawn_slot_free(route, cars, s_spawn: float, min_gap: float = 25.0, car_len: float = 4.5) -> bool:
    """Return True if there is enough headway to spawn at s_spawn on route."""
    L = _route_len(route)
    if L <= 0.0:
        return False
    for c in cars:
        if getattr(c, "route", None) is route:
            ahead = (c.s - s_spawn) % L
            if ahead <= min_gap:          # a car is too close ahead
                return False
            if (L - ahead) <= car_len:    # a car just passed (too close behind)
                return False
    return True

def _xy_of_route(r, s: float) -> Tuple[float, float]:
    """Return (x, y) at distance s along route r. Works with .pos_at or .xy_at and
    normalizes tuples/lists/np arrays or geometry-like objects with .x/.y."""
    pos_at = getattr(r, "pos_at", None)
    xy_at  = getattr(r, "xy_at", None)

    fn = pos_at if callable(pos_at) else (xy_at if callable(xy_at) else None)
    if fn is None:
        raise AttributeError("Route has neither pos_at nor xy_at.")

    pt = fn(float(s))

    if hasattr(pt, "x") and hasattr(pt, "y"):
        return float(getattr(pt, "x")), float(getattr(pt, "y"))

    try:
        seq = cast(Sequence[Any], pt)
        return float(seq[0]), float(seq[1])
    except Exception:
        pass

    try:
        return float(pt["x"]), float(pt["y"])  # type: ignore[index]
    except Exception:
        pass

    raise TypeError("Expected point-like return (x,y), object with .x/.y, or mapping with 'x'/'y'.")


# ---------------- Renderer ----------------
class Renderer(ap.Model):
    """AgentPy model + Matplotlib renderer. Heuristic controls lights. No pair constraints."""

    def setup(self):
        # --- params ---
        p = self.p
        self.dt = float(p.get('dt', 1.0))

        # Container for cars
        self.cars = ap.AgentList(self, 0, Car)

        # Give each TL a stable id if missing
        for i, tl in enumerate(traffic_lights):
            if getattr(tl, "id", None) in (None, ""):
                tl.id = f"TL_{i}"

        # --- figure/axes ---
        self.fig: Figure = plt.figure(figsize=(9, 9))  # type: ignore
        self.ax: Axes = self.fig.add_subplot(1, 1, 1)  # type: ignore
        self.ax.set_aspect('equal', adjustable='box')

        # Draw routes and traffic lights
        self._plot_routes_background()
        for tl in traffic_lights:
            tl.plot(self.ax)

        # Bounds
        self._auto_bounds()

        # --- stochastic spawner (to avoid periodic sync) ---
        self.rng = np.random.default_rng()
        self.MAX_CARS = int(p.get("MAX_CARS", 80))               # global cap
        self.HEADWAY_MEAN = float(p.get("HEADWAY_MEAN", 12.0))   # avg seconds between arrivals per route
        self.SPAWN_S = float(p.get("SPAWN_S", 0.0))              # spawn position (s) on each route
        self.MIN_GAP = float(p.get("MIN_GAP", 25.0))             # forward gap required to spawn
        self.CAR_LEN = float(p.get("CAR_LEN", 4.5))              # overlap tolerance behind
        self.SPEED_MIN = float(p.get("SPEED_MIN", 3.5))
        self.SPEED_MAX = float(p.get("SPEED_MAX", 6.5))

        # one ETA per route index (avoid using Route as dict key)
        self._spawn_eta = [0.0] * len(routes)
        for i, _r in enumerate(routes):
            jitter = float(self.rng.uniform(0.0, self.HEADWAY_MEAN))
            self._spawn_eta[i] = float(self.rng.exponential(self.HEADWAY_MEAN) + jitter)

        # live updates
        try:
            plt.ion()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    # ---------- plotting ----------
    def _plot_routes_background(self):
        for i, r in enumerate(routes):
            color = route_colors[i] if i < len(route_colors) else 'gray'
            width = route_widths[i] if i < len(route_widths) else 2.0
            L = _route_len(r)
            if L <= 0:
                continue
            s_vals = np.linspace(0.0, L, 150, dtype=float)
            xs, ys = [], []
            for s in s_vals:
                x, y = _xy_of_route(r, float(s))
                xs.append(x); ys.append(y)
            self.ax.plot(xs, ys, '-', color=color, linewidth=width, alpha=0.85, zorder=1)
            if i < len(route_names):
                xm, ym = _xy_of_route(r, 0.5 * L)
                self.ax.text(xm, ym, route_names[i], fontsize=7, color=color, alpha=0.7)

    def _auto_bounds(self):
        xs, ys = [], []
        for r in routes:
            L = _route_len(r)
            if L <= 0:
                continue
            for s in np.linspace(0.0, L, 300, dtype=float):
                x, y = _xy_of_route(r, float(s))
                xs.append(x); ys.append(y)
        if xs and ys:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            pad = 0.05 * max(xmax - xmin, ymax - ymin, 1.0)
            self.ax.set_xlim(xmin - pad, xmax + pad)
            self.ax.set_ylim(ymin - pad, ymax + pad)
            self.ax.grid(True, alpha=0.2, zorder=0)

    def plot(self):
        for car in self.cars:
            if hasattr(car, "plot"):
                car.plot(self.ax)
        for tl in traffic_lights:
            if hasattr(tl, "patch") and tl.patch is not None:
                tl.patch.set_zorder(30)
        try:
            self.fig.canvas.draw_idle()      # type: ignore
            self.fig.canvas.flush_events()   # type: ignore
            plt.pause(0.001)
        except Exception:
            pass

    # ---------- convenience ----------
    def spawn_car(self, route, s: float | None = None, ds: float | None = None, **kwargs):
        """Create and register a car on a given route; returns the car."""
        conns = [tlc for tlc in tlconnections if tlc.route is route]
        cid = len(self.cars)
        q = getattr(self, "q_learner", None)
        car = Car(self, route, conns, cid, q)
        if s is not None:
            car.s = float(s)
            car.position = _xy_of_route(route, car.s)
        if ds is not None:
            car.ds = float(ds)
        self.cars.append(car)
        if hasattr(car, "plot"):
            car.plot(self.ax)
        return car
    

    def get_queues_by_traffic_light(self, cars, tlconnections, threshold: float = 200.0):
        """
        Returns a dict {traffic_light_id: queue_len}.
        A car counts toward the queue if:
        - it is on the same route as the TLConnection,
        - it is before the stop line (by s),
        - it is 'stopped' because of the light (car.state == 'stop'),
        - and its distance to the stop line is <= threshold.
        Only queues for RED lights are counted; GREEN/YELLOW -> 0.
        """
        from collections import defaultdict
        from traffic_light import TrafficLightState

        # Group tlconnections by light id
        tlc_by_light = defaultdict(list)
        for tlc in tlconnections:
            tl = getattr(tlc, "traffic_light", None)
            tl_id = getattr(tl, "id", None)
            if tl_id is not None:
                tlc_by_light[tl_id].append(tlc)

        queues = {}

        for tl_id, tlc_list in tlc_by_light.items():
            tl = tlc_list[0].traffic_light
            # Only measure queues when the light is RED
            if tl.state != TrafficLightState.RED:
                queues[tl_id] = 0
                continue

            count_queue = 0
            for tlc in tlc_list:
                route = tlc.route
                L = float(getattr(route, "length", 0.0) or 0.0)

                # Cars on the same route, located BEFORE the stop line
                cars_on_route = [c for c in cars
                                if getattr(c, "route", None) is route
                                and getattr(c, "s", 0.0) <= tlc.s]

                # Sort nearest to the stop first
                cars_on_route.sort(key=lambda c: (tlc.s - c.s))

                # Count consecutive stopped cars within the threshold window
                for c in cars_on_route:
                    ahead = (tlc.s - c.s) if L <= 0.0 else ((tlc.s - c.s) % L)
                    if getattr(c, "state", None) == "stop" and ahead <= threshold:
                        count_queue += 1
                    else:
                        # Queue stops at the first car that is not stopped / too far
                        break

            queues[tl_id] = count_queue

        return queues


    # ---------- main tick ----------
    def step(self):


        # --- stochastic arrivals (Poisson-like per route) ---
        dt = self.dt if hasattr(self, "dt") else 1.0
        if len(self.cars) < self.MAX_CARS:
            for i, r in enumerate(routes):
                L = _route_len(r)
                if L <= 0.0:
                    continue
                self._spawn_eta[i] -= dt
                while self._spawn_eta[i] <= 0.0 and len(self.cars) < self.MAX_CARS:
                    s0 = (self.SPAWN_S + float(self.rng.uniform(0.0, 2.5))) % L
                    if _spawn_slot_free(r, self.cars, s0, self.MIN_GAP, self.CAR_LEN):
                        ds = float(self.rng.uniform(self.SPEED_MIN, self.SPEED_MAX))
                        self.spawn_car(r, s=s0, ds=ds)
                    self._spawn_eta[i] += float(self.rng.exponential(self.HEADWAY_MEAN))

        # --- Map TL -> its TLConnections ---
        tl_to_conns = {}
        for tlc in tlconnections:
            tl = getattr(tlc, "traffic_light", None)
            if tl is None:
                continue
            tl_to_conns.setdefault(tl, []).append(tlc)

        # --- Compute queues for each TL (max cars within 'near' of any approach stop line) ---
        q_by_tl = {}
        near = 60.0
        for tl, conns in tl_to_conns.items():
            queue = 0
            for tlc in conns:
                r = getattr(tlc, "route", None)
                if r is None:
                    continue
                L = _route_len(r)
                if L <= 1e-9:
                    continue
                cnt = 0
                for c in self.cars:
                    if getattr(c, "route", None) is r:
                        ahead = (tlc.s - c.s) % L
                        if 0.0 <= ahead <= near:
                            cnt += 1
                if cnt > queue:
                    queue = cnt
            q_by_tl[tl] = queue

        # --- Apply heuristic for each TL (no pair constraints, no gating) ---
        tl_by_id = {getattr(tl, "id", f"tl_{i}"): tl for i, tl in enumerate(traffic_lights)}
        # Compute per-light queues based on stopped cars near each stop line
        queues_by_id = self.get_queues_by_traffic_light(self.cars, tlconnections, threshold=200.0)

        for tl in traffic_lights:
            tl_id = getattr(tl, "id", None)
            q = queues_by_id.get(tl_id, 0)

            if hasattr(tl, "heuristic_control"):
                tl.heuristic_control(
                    queue=q,
                    autos_en_rotonda=0,
                    pairs=[],                   # keep whatever you use here
                    tl_by_id={},
                    params={},                  # use defaults in TrafficLight
                    cars=self.cars,
                    tlconnections=tlconnections,
                    can_turn_green=True,
                )

            if hasattr(tl, "step"):
                tl.step(self.dt if hasattr(self, "dt") else 1.0)

        # --- Move cars AFTER lights decided their states ---
        for car in self.cars:
            if hasattr(car, "step"):
                car.step()

        # --- Redraw frame ---
        self.plot()

    def after(self):
        try:
            plt.ioff()
            plt.show()
        except Exception:
            pass
