# model.py
import agentpy as ap
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches
import json, math
from route import Route
from car import Car
from defined_routes import routes, route_colors, route_widths, route_names
from defined_tlconnections import tlconnections, traffic_lights
from network_manager import NetManager
import math
from typing import Any, Dict  # put at top of file
import time
from network_manager import NetManager
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
        id = 1
        for r in self.routes:
            conns = [tlc for tlc in tlconnections if tlc.route is r]
            self.cars.append(Car(self, r, conns, id))
            id+=1

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

    # --- add this helper in your model class ---
    def _ensure_net(self):
        if not hasattr(self, "net") or self.net is None:
            self.net = NetManager(host="127.0.0.1", port=8080)
            self.net.start()
        # real-time timing init (run once)
        if not hasattr(self, "_rt_inited"):
            self._dt = 0.05            # 20 Hz snapshots; change to 0.1 for 10 Hz, etc.
            self._tick = 0
            self._t0 = time.perf_counter()
            self._rt_inited = True
    def step(self):
        # 1) Avanza coches
        for car in getattr(self, "cars", []):
            if hasattr(car, "step"):
                car.step()
        # 2) Avanza semáforos si tienes objetos con step()
        for tl in (globals().get("traffic_lights") or []):
            if hasattr(tl, "step"):
                tl.step()
        # 3) (Opcional) plot si quieres conservarlo
        if hasattr(self, "plot"):
            self.plot()
        # 4) Empuja estado
        self.push_state()

    def push_state(self):
        self._ensure_net()

        def car_row(car):
            # position from car.position or route + s
            if hasattr(car, "position") and car.position is not None:
                x = float(car.position[0]); y = float(car.position[1])
            else:
                px, py = car.route.pos_at(float(getattr(car, "s", 0.0)))
                x = float(px); y = float(py)

            # heading in DEGREES for Unity
            try:
                hdg_rad = float(car.heading())
            except Exception:
                hdg_rad = 0.0
            hdg_deg = math.degrees(hdg_rad)

            car_id = int(getattr(car, "id", id(car)))
            route_obj = getattr(car, "route", None)
            route_id = int(getattr(route_obj, "id", id(route_obj)))

            return {
                "id": car_id,
                "position": [x, y],
                "heading": float(hdg_deg),
                "route_id": route_id,
                # extras (Unity can ignore them)
                "speed": float(getattr(car, "ds", 0.0)),
                "s": float(getattr(car, "s", 0.0)),
                "route_name": getattr(route_obj, "name", "") or "",
                "length": float(getattr(car, "length", 4.2)),
                "width": float(getattr(car, "width", 1.8)),
            }

        cars_payload = [car_row(c) for c in getattr(self, "cars", []) if getattr(c, "active", True)]

        # ---- Lights payload for floor pads (safe if you don't have lights yet) ----
        lights_payload: list[Dict[str, Any]] = []

        # Prefer a model attribute; change this to your source of lights if different.
        traffic_lights_iter = getattr(self, "traffic_lights", None) or (globals().get("traffic_lights") or [])


        for tl in traffic_lights_iter:
            # ID
            lid = str(getattr(tl, "name", getattr(tl, "id", "TL")))

            # State -> string (supports enum-like objects and plain strings)
            state_val: Any = getattr(tl, "state", "RED")
            state_str = str(getattr(state_val, "name", state_val))
            

            # Start row as Dict[str, Any] so floats are allowed
            row: Dict[str, Any] = {"light_id": lid, "state": state_str}

            # Optional coordinates
            x = getattr(tl, "x", None)
            y = getattr(tl, "y", None)
            if x is not None and y is not None:
                row["x"] = float(x)
                row["y"] = float(y)

            # Optional rotation
            rot = getattr(tl, "rotation_deg", None)
            row["rotation_deg"] = float(rot) if rot is not None else 0.0

            lights_payload.append(row)

        # Send both; Unity is fine if "lights" is empty
        state = {"cars": cars_payload, "lights": lights_payload}
        self.net.push_state(state)  # pass dict; NetManager will json.dumps once
    
    def get_car_states(self):
        return [car.export_state() for car in self.cars]
    
    @staticmethod
    def discretizar_distancias(distancia, num_buckets=5):
        if distancia == 0:
            distancia = 0
        elif distancia >= 10: 
            distancia = 10
        
        t_bucket = 10  / num_buckets
        bucket = distancia // t_bucket

        if bucket == num_buckets:
            bucket -= 1
        return bucket

    def get_state(self, car: Car):
        """
        Computes the discretized state for a car agent.
        The state is a tuple containing:
        - Discretized distance to the car in front.
        - Discretized speed of the car.
        - State of the next traffic light (0: Green, 1: Yellow, 2: Red).
        - Discretized distance to the next traffic light.
        """
        # 1. Distancia al coche de enfrente
        car_en_frente_dist = 10.0  # Valor por defecto si no hay coche enfrente
        for other_car in self.cars:
            if other_car is not car and other_car.route is car.route and other_car.s > car.s:
                dist = other_car.s - car.s
                if dist < car_en_frente_dist:
                    car_en_frente_dist = dist
        
        distancia_discreta_coche = self.discretizar_distancias(car_en_frente_dist)

        # 2. Velocidad actual del coche
        velocidad_discreta = self.discretizar_distancias(car.ds)

        # 3. Estado y distancia al próximo semáforo
        next_tlc = car.nextTLC
        if next_tlc:
            # Mapeo del estado del semáforo a un entero
            estado_semaforo_map = {
                "GREEN": 0,
                "YELLOW": 1,
                "RED": 2
            }
            estado_semaforo = estado_semaforo_map.get(next_tlc.traffic_light.state.name, 2) # Default to RED
            
            # Distancia al semáforo a lo largo de la ruta
            distancia_semaforo = (next_tlc.s - car.s) % car.route.length
            distancia_discreta_semaforo = self.discretizar_distancias(distancia_semaforo)
        else:
            # Valores por defecto si no hay semáforo en la ruta
            estado_semaforo = 0  # Como si estuviera siempre en verde
            distancia_discreta_semaforo = self.discretizar_distancias(10.0) # Distancia máxima

        return (distancia_discreta_coche, velocidad_discreta, estado_semaforo, distancia_discreta_semaforo)
    
