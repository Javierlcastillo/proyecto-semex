# defined_routes.py
import json, os
from route import Route

ROUTES_PATH = "datos/routes.json"
META_PATH   = ROUTES_PATH.replace(".json", "_meta.json")

# tweak if you want thicker/thinner lines in Matplotlib
WIDTH_SCALE = 6.0
MIN_WIDTH   = 1.0

routes: list[Route] = []
route_colors: list[str] = []
route_widths: list[float] = []
route_names: list[str] = []

# geometry
with open(ROUTES_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for points in data:
    routes.append(Route.from_polyline(points, samples_per_segment=50))

# styles (optional meta)
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        metas = json.load(f)

    for m in metas:
        route_colors.append(m.get("color", "#000000"))
        w = float(m.get("width", 0.15)) * WIDTH_SCALE
        route_widths.append(w if w >= MIN_WIDTH else MIN_WIDTH)
        route_names.append(m.get("name") or "")
else:
    # fallback palette if meta isn't present
    fallback = ['#FFD200', '#FF3B30', '#34C759', '#5856D6']
    for i in range(len(routes)):
        route_colors.append(fallback[i % len(fallback)])
        route_widths.append(2.0)
        route_names.append(f"route_{i}")
