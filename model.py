# model.py
import agentpy as ap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches

from route import Route
from car import Car
from defined_routes import routes, route_colors, route_widths
from defined_tlconnections import tlconnections


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
        # if it's callable (e.g., route.length()), try calling
        try:
            if callable(v):
                vv = v()
            else:
                vv = v
            return float(vv)  # type: ignore[arg-type]
        except Exception:
            continue
    return None

def _xy_from_point(p):
    """Return (x, y) from many shapes: (x,y), numpy array, or object with .x/.y."""
    try:
        return float(p[0]), float(p[1])  # type: ignore[index]
    except Exception:
        if hasattr(p, "x") and hasattr(p, "y"):
            return float(getattr(p, "x")), float(getattr(p, "y"))
        if hasattr(p, "X") and hasattr(p, "Y"):
            return float(getattr(p, "X")), float(getattr(p, "Y"))
        # last resort: coerce to numpy, flatten
        a = np.asarray(p).ravel()
        return float(a[0]), float(a[1])

def _xy_list(pts):
    out = []
    for p in pts:
        out.append(_xy_from_point(p))
    return out

def _get_polyline(route: Route):
    """
    Return a list of (x, y) tuples for plotting, tolerant of different route implementations:
      - route.polyline / .points / .pts / .waypoints
      - sampling with route.pos_at(...) using a length (attr or method)
    """
    # 1) Direct containers
    for name in ("polyline", "points", "pts", "waypoints"):
        pts = getattr(route, name, None)
        if pts is not None:
            try:
                return _xy_list(pts)
            except Exception:
                pass  # fall through to other options

    # 2) Sample using pos_at if available
    pos_at = _callable_attr(route, ("pos_at", "posAt", "point_at", "pointAt", "position_at", "positionAt", "eval"))
    if pos_at:
        L = _numeric_or_callable_value(route, ("length", "L", "total_length", "arc_length", "arclength", "len"))
        if L is None:
            L = 200.0  # fallback
        n = max(50, int(L * 2))
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

    # 3) Give up
    return []


# ---------- Model ----------

class Renderer(ap.Model):
    """
    Renders routes, traffic lights and cars.
    - Routes are plotted here (we don't call Route.plot).
    - Colors/widths come from defined_routes (exported from Unity).
    """

    fig: Figure
    ax: Axes

    routes: list[Route]
    car_patches: list[patches.Rectangle] = []

    def setup(self):
        # Data
        self.routes = routes
        self.cars = [Car(self, r, [tlc for tlc in tlconnections if tlc.route == r]) for r in self.routes]

        # Figure / Axes
        self.fig, self.ax = plt.subplots(figsize=(8, 8), squeeze=True)  # type: ignore
        self.ax.set_aspect("equal")
        self.ax.set_xlim(100, 700)
        self.ax.set_ylim(0, 600)

        # Plot routes directly
        default_palette = [
            "#FFD200", "#FF3B30", "#34C759", "#5856D6",
            "#A2845E", "#00BCD4", "#FF2D55", "#1E90FF"
        ]

        for i, r in enumerate(self.routes):
            c = route_colors[i] if i < len(route_colors) else default_palette[i % len(default_palette)]
            lw = route_widths[i] if i < len(route_widths) else 2.0

            pts = _get_polyline(r)
            if not pts:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self.ax.plot(xs, ys, color=c, linewidth=lw)

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
        # Advance cars, then repaint
        for car in self.cars:
            if hasattr(car, "step"):
                car.step()
        self.plot()
