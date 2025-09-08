# defined_tlconnections.py
from __future__ import annotations
import json, os
from typing import Dict, List, Tuple
import numpy as np

from defined_routes import routes, route_names
from traffic_light import TrafficLight, TrafficLightState, TLConnection

# ---- locate file ------------------------------------------------------------
CANDIDATES = (
    "traffic_lights.json",
    "data/traffic_lights.json",
    "datos/traffic_lights.json",
)
PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if PATH is None:
    raise FileNotFoundError(f"traffic_lights.json not found. Tried: {', '.join(CANDIDATES)}")

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- build TrafficLight objects --------------------------------------------
traffic_lights: List[TrafficLight] = []
lights_by_id: Dict[str, TrafficLight] = {}

for i, L in enumerate(data.get("lights", [])):
    lid = str(L.get("id") or f"tl_{i}")
    x, y = float(L["pos"][0]), float(L["pos"][1])
    rot  = float(L.get("rotation", 0.0))
    tl = TrafficLight(x, y, rot)

    st = (L.get("state") or "RED").upper()
    if st in TrafficLightState.__members__:
        tl.state = TrafficLightState[st]
    traffic_lights.append(tl)
    lights_by_id[lid] = tl

# ---- helper: nearest arclength "s" on a route to a point --------------------
def nearest_s(route, x: float, y: float, samples: int = 1500) -> Tuple[float, float]:
    """Returns (s, euclidean_dist) for the closest sampled point on `route` to (x,y)."""
    L = float(getattr(route, "length", 0.0))
    if L <= 0:
        return 0.0, float("inf")
    svals = np.linspace(0.0, L, samples)
    pts   = np.array([route.pos_at(float(s)) for s in svals])   # (N,2)
    d2    = (pts[:, 0] - x)**2 + (pts[:, 1] - y)**2
    k     = int(np.argmin(d2))
    return float(svals[k]), float(np.sqrt(d2[k]))

# ---- build TLConnection list -----------------------------------------------
tlconnections: List[TLConnection] = []

# 1) If your JSON already contains explicit connections, honor them:
name2route = {
    (getattr(r, "name", "") or (route_names[i] if i < len(route_names) else f"route_{i}")): r
    for i, r in enumerate(routes)
}

explicit = False
for C in data.get("connections", []):
    explicit = True
    lid = str(C.get("light", ""))
    tl  = lights_by_id.get(lid)
    if tl is None:
        print(f"[WARN] No light with id={lid!r}; skipping connection {C}")
        continue

    r = None
    if "route" in C:
        r = name2route.get(str(C["route"]))
    elif "route_index" in C:
        idx = int(C["route_index"])
        if 0 <= idx < len(routes):
            r = routes[idx]

    if r is None:
        print(f"[WARN] Route not found for {C}; skipping.")
        continue

    s = float(C.get("s", 0.0))
    s = max(0.0, min(s, getattr(r, "length", 0.0)))
    tlconnections.append(TLConnection(r, tl, s))

# 2) If there were no explicit connections, auto-derive them by proximity:
if not explicit:
    DIST_THR = 25.0  # pixels/units in the Python coordinate frame (tweak if needed)
    for tl in traffic_lights:
        for r in routes:
            s, dist = nearest_s(r, tl.x, tl.y)
            if dist <= DIST_THR:
                tlconnections.append(TLConnection(r, tl, s))

__all__ = ["traffic_lights", "tlconnections", "PATH"]
