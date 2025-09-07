# defined_routes.py
import json, os
from route import Route

# Try a few common locations; tweak if yours is different.
_CANDIDATES = [
    "datos/routes.json",
    "routes.json",
    "data/routes.json",
]

def _find_existing(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "routes.json not found. Tried: " + ", ".join(cands)
    )

ROUTES_PATH = _find_existing(_CANDIDATES)
META_PATH   = ROUTES_PATH.replace(".json", "_meta.json")

# Scale linewidths from Unity a bit for Matplotlib aesthetics
WIDTH_SCALE = 6.0
MIN_WIDTH   = 1.0

routes = []          # list[Route]
route_colors = []    # list[str]  hex like "#FF0000"
route_widths = []    # list[float]
route_names  = []    # list[str]

# ---- geometry
with open(ROUTES_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if not hasattr(Route, "from_polyline"):
    raise RuntimeError("Route.from_polyline(...) not found. "
                       "Your Route class must provide it for this loader.")

for poly in data:
    # poly is [[x,y], ...]
    routes.append(Route.from_polyline(poly, samples_per_segment=50))

# ---- styles (optional meta)
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        metas = json.load(f)
    for m in metas:
        route_colors.append(m.get("color", "#000000"))
        w = float(m.get("width", 0.15)) * WIDTH_SCALE
        route_widths.append(max(MIN_WIDTH, w))
        route_names.append(m.get("name") or "")
else:
    # fallback palette if meta isn't present
    fallback = ['#FFD200', '#FF3B30', '#34C759', '#5856D6',
                '#A2845E', '#00BCD4', '#FF2D55', '#1E90FF']
    for i in range(len(routes)):
        route_colors.append(fallback[i % len(fallback)])
        route_widths.append(2.0)
        route_names.append(f"route_{i}")
