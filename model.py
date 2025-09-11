# model.py
import agentpy as ap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches

from route import Route
from car import Car
from defined_routes import routes, route_colors, route_widths, route_names
from defined_tlconnections import tlconnections, traffic_lights

# ---------- helpers (robust + Pylance-friendly) ----------

def _callable_attr(obj, names):
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn
    return None

def _numeric_or_callable_value(obj, names):
    """Return float value from an attribute or 0-arg method if possible."""
    for n in names:
        v = getattr(obj, n, None)
        if v is None:
            continue
        try:
            vv = v() if callable(v) else v
            return float(vv)  # type: ignore[arg-type]
        except Exception:
            continue
    return None

def _xy_from_point(p):
    """Accept (x,y) lists/tuples/np arrays or objects with .x/.y/.X/.Y."""
    try:
        return float(p[0]), float(p[1])  # type: ignore[index]
    except Exception:
        if hasattr(p, "x") and hasattr(p, "y"):
            return float(getattr(p, "x")), float(getattr(p, "y"))
        if hasattr(p, "X") and hasattr(p, "Y"):
            return float(getattr(p, "X")), float(getattr(p, "Y"))
        a = np.asarray(p).ravel()
        return float(a[0]), float(a[1])

def _xy_list(pts):
    return [_xy_from_point(p) for p in pts]

def _get_polyline(route: Route):
    """
    Return a list of (x,y) for plotting. Tries common containers first,
    then samples using pos_at() + length if needed.
    """
    for name in ("polyline", "points", "pts", "waypoints"):
        pts = getattr(route, name, None)
        if pts is not None:
            try:
                return _xy_list(pts)
            except Exception:
                pass

    pos_at = _callable_attr(route, ("pos_at", "posAt", "point_at", "pointAt",
                                    "position_at", "positionAt", "eval"))
    if pos_at:
        L = _numeric_or_callable_value(route, ("length", "L", "total_length",
                                               "arc_length", "arclength", "len"))
        if L is None:
            L = 200.0  # fallback
        n = max(50, int(float(L) * 2))
        ss = np.linspace(0.0, float(L), n)
        pts = []
        for s in ss:
            try:
                pt = pos_at(float(s))  # type: ignore[misc]
                pts.append(_xy_from_point(pt))
            except Exception:
                continue
        if pts:
            return pts

    return []


# ---------- Model ----------

class Renderer(ap.Model):
    """Renders routes, traffic lights and cars."""

    fig: Figure
    ax: Axes

    routes: list[Route]
    car_patches: list[patches.Rectangle] = []

    def setup(self):
        # Data
        self.routes = routes

        # Build cars with TL connections (your code here)
        self.cars = []
        for r in self.routes:
            conns = [tlc for tlc in tlconnections if tlc.route is r]
            cid = len(self.cars)  # simple unique id
            q = getattr(self, "q_learner", None)
            self.cars.append(Car(self, r, conns, cid, q))

        # Figure / Axes
        self.fig, self.ax = plt.subplots(figsize=(8, 8), squeeze=True)  # type: ignore
        self.ax.set_aspect("equal")

        # draws the small rectangle at (tl.x, tl.y) with tl.rotation    
        for tl in traffic_lights:
            tl.plot(self.ax)
            

        # --- Compute bounds from the actual routes and zoom out automatically ---

        def _bounds(rs, samples_per_route: int = 400):
            xs, ys = [], []
            for rt in rs:
                L = getattr(rt, "length", 0.0)
                if L <= 0:
                    continue
                s_vals = np.linspace(0.0, L, samples_per_route)
                for s in s_vals:
                    x, y = rt.pos_at(float(s))
                    xs.append(x); ys.append(y)
            if not xs:   # fallback
                return (0, 1, 0, 1)
            return (min(xs), max(xs), min(ys), max(ys))

        xmin, xmax, ymin, ymax = _bounds(self.routes)
        pad = 0.10 * max(xmax - xmin, ymax - ymin)  # 10% padding
        self.ax.set_xlim(xmin - pad, xmax + pad)
        self.ax.set_ylim(ymin - pad, ymax + pad)

        # (Optional) debug overlay to see starts/ends + current car positions
        for i, r in enumerate(self.routes):
            x0, y0 = r.pos_at(0.0)
            xL, yL = r.pos_at(getattr(r, "length", 0.0))
            self.ax.scatter([x0, xL], [y0, yL], s=40, edgecolors="k", zorder=5)
        for c in self.cars:
            x, y = c.position
            self.ax.scatter([x], [y], s=60, facecolors="none", edgecolors="k", zorder=6)


        # Plot routes directly (colors/widths from Unity meta)
        default_palette = [
            "#FFD200", "#FF3B30", "#34C759", "#5856D6",
            "#A2845E", "#00BCD4", "#FF2D55", "#1E90FF"
        ]

        for i, r in enumerate(self.routes):
            c  = route_colors[i] if i < len(route_colors) else default_palette[i % len(default_palette)]
            lw = route_widths[i] if i < len(route_widths) else 2.0

            pts = _get_polyline(r)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self.ax.plot(xs, ys, color=c, linewidth=lw)

            
        for i, r in enumerate(routes):
            print(i, getattr(r, "name", f"route_{i}"), r.length)
        # Traffic lights
        for tlc in tlconnections:
            if hasattr(tlc, "traffic_light") and hasattr(tlc.traffic_light, "plot"):
                tlc.traffic_light.plot(self.ax)

        self.fig.show()

    def plot(self):
        # Draw cars at current positions
        for car in self.cars:
            if hasattr(car, "plot"):
                car.plot(self.ax)
        self.fig.canvas.draw_idle()      # type: ignore
        self.fig.canvas.flush_events()   # type: ignore

    def step(self):
        # --- Map each traffic light to its TLConnections (to measure its queue) ---
        tl_to_conns = {}
        for tlc in tlconnections:
            tl = getattr(tlc, "traffic_light", None)
            if tl is None:
                continue
            tl_to_conns.setdefault(tl, []).append(tlc)

        # If you ever enforce conflicts, fill this with (id_a, id_b) pairs
        pairs = []  # e.g., [("TL_N","TL_S"), ("TL_E","TL_W")]
        tl_by_id = {getattr(tl, "id", f"tl_{i}"): tl for i, tl in enumerate(traffic_lights)}

        # --- 1) Apply the heuristic controller to each light (decide state this tick) ---
        for tl in traffic_lights:
            conns = tl_to_conns.get(tl, ())
            # queue = max cars approaching this light (within 'near' meters of its stop line)
            queue = 0
            for tlc in conns:
                r = getattr(tlc, "route", None)
                if r is None:
                    continue
                L = float(getattr(r, "length", 0.0) or 0.0)
                if L <= 1e-9:
                    continue
                near = 60.0
                cnt = 0
                for c in self.cars:
                    if getattr(c, "route", None) is r:
                        # distance along route from the car to the stop position
                        ahead = (tlc.s - c.s) % L
                        if 0.0 <= ahead <= near:
                            cnt += 1
                if cnt > queue:
                    queue = cnt

            # Call your heuristic (it sets tl.state). Empty params -> defaults inside TrafficLight.
            if hasattr(tl, "heuristic_control"):
                tl.heuristic_control(
                    queue=queue,
                    autos_en_rotonda=0,
                    pairs=pairs,
                    tl_by_id=tl_by_id,
                    params={},                    # use defaults declared in TrafficLight
                    cars=self.cars,
                    tlconnections=tlconnections,
                    can_turn_green=True,
                )

            # Update the visual (your tl.step should only redraw, not auto-cycle)
            if hasattr(tl, "step"):
                tl.step(self.dt if hasattr(self, "dt") else 1.0)

        # --- 2) Move cars after lights decided their states ---
        for car in self.cars:
            if hasattr(car, "step"):
                car.step()

        # --- 3) Redraw frame ---
        self.plot()




