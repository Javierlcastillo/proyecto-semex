# defined_routes.py
import json
import os
from typing import List, Tuple, Any
from route import Route

# --- Where to look for the JSON files
_CANDIDATES = [
    "datos/routes.json",
    "routes.json",
    "data/routes.json",
]

def _find_existing(cands: List[str]) -> str:
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("routes.json not found. Tried: " + ", ".join(cands))

ROUTES_PATH = _find_existing(_CANDIDATES)

# Accept either "<name>_meta.json" or "routes_meta.json" in the same folder
_base, _ext = os.path.splitext(ROUTES_PATH)
META_PATHS = [f"{_base}_meta.json", os.path.join(os.path.dirname(ROUTES_PATH), "routes_meta.json")]

# --- Style defaults
WIDTH_SCALE = 4.0   # scale widths (e.g., Unity->Matplotlib aesthetics)
MIN_WIDTH   = 1.0
FALLBACK_COLORS = ['#FFD200', '#FF3B30', '#34C759', '#5856D6',
                   '#A2845E', '#00BCD4', '#FF2D55', '#1E90FF']

# --- Public objects you can import elsewhere
routes: List[Route] = []
route_colors: List[str] = []
route_widths: List[float] = []
route_names:  List[str] = []

# ---- Helpers ----------------------------------------------------------------

def _normalize_poly(points: Any) -> List[Tuple[float, float]]:
    """
    Accepts various shapes (list of [x,y], list of tuples, etc.) and returns
    a clean list[(x,y)] as floats.
    """
    out = []
    for pt in points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x, y = pt[0], pt[1]
            out.append((float(x), float(y)))
        else:
            raise ValueError("Invalid point format in polyline: expected [x, y].")
    if not out:
        raise ValueError("Empty polyline.")
    return out

def _extract_routes_from_json(data: Any) -> List[List[Tuple[float, float]]]:
    """
    Supports:
      1) {"routes": [{"points": [[x,y],...], "name":..., "color":..., "width":...}, ...]}
      2) [{"points": [[x,y],...], ...}, ...]
      3) [ [[x,y],...], [[x,y],...], ... ]
      4) Single polyline [[x,y], ...]  (wrapped into a list)
    Also returns per-item meta if present via the caller (we read meta below as a separate file).
    """
    polys: List[List[Tuple[float, float]]] = []

    if isinstance(data, dict) and "routes" in data:
        for item in data["routes"]:
            pts = item.get("points") or item.get("coords") or item.get("polyline") or item.get("path")
            if pts is None:
                raise ValueError("Route entry missing 'points/coords/polyline/path'.")
            polys.append(_normalize_poly(pts))
    elif isinstance(data, list):
        if not data:
            return []
        # list of dicts?
        if isinstance(data[0], dict):
            for item in data:
                pts = item.get("points") or item.get("coords") or item.get("polyline") or item.get("path")
                if pts is None:
                    raise ValueError("Route entry missing 'points/coords/polyline/path'.")
                polys.append(_normalize_poly(pts))
        else:
            # Could be a list of polylines OR a single polyline
            # Heuristic: if the first element is [x,y] (numbers), treat as single polyline
            first = data[0]
            is_xy = isinstance(first, (list, tuple)) and len(first) >= 2 and all(
                isinstance(first[k], (int, float)) for k in (0, 1)
            )
            if is_xy:
                polys.append(_normalize_poly(data))  # single polyline
            else:
                # assume list of polylines
                for pts in data:
                    polys.append(_normalize_poly(pts))
    else:
        raise ValueError("Unsupported routes.json structure.")
    return polys

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _try_build_route(poly: List[Tuple[float, float]]) -> Route:
    # Some projects have a different signature; try both.
    try:
        return Route.from_polyline(poly, samples_per_segment=50)
    except TypeError:
        return Route.from_polyline(poly)

def _load_meta() -> List[dict]:
    for mp in META_PATHS:
        if os.path.exists(mp):
            metas = _load_json(mp)
            if isinstance(metas, dict) and "routes" in metas:
                metas = metas["routes"]
            if not isinstance(metas, list):
                raise ValueError("Meta file must be a list or an object with 'routes'.")
            return metas
    return []  # no meta file found

def _dedup(points, eps=1e-6):
    out = [points[0]]
    for x, y in points[1:]:
        if (x - out[-1][0])**2 + (y - out[-1][1])**2 > eps*eps:
            out.append((x, y))
    return out

# ---- Build routes + styles ---------------------------------------------------

_data = _load_json(ROUTES_PATH)
_polys = _extract_routes_from_json(_data)
_metas = _load_meta()

for i, poly in enumerate(_polys):
    poly = _dedup(poly)
    r = _try_build_route(poly)

    # Save the exact first coordinate from the JSON so cars can spawn there.
    # This is the raw point BEFORE any resampling that Route might do.
    x0, y0 = poly[0]          # works if each point is [x, y] or (x, y)
    r.json_start = (float(x0), float(y0))

    # Optional: attach a human-readable name if later useful (from meta or fallback)
    name = ""
    if i < len(_metas):
        name = _metas[i].get("name", "") or ""
    if not name:
        name = f"route_{i}"
    r.name = name  # harmless attribute assignment on the Route instance

    routes.append(r)

# Styles: color/width/name arrays aligned with routes
for i in range(len(routes)):
    if i < len(_metas):
        m = _metas[i]
        color = m.get("color", None)
        width = m.get("width", None)
    else:
        m = {}
        color = None
        width = None

    route_colors.append(color if color else FALLBACK_COLORS[i % len(FALLBACK_COLORS)])

    # Width: allow numeric, else fallback (scaled and clamped)
    if width is None:
        route_widths.append(2.0)
    else:
        try:
            w = float(width) * WIDTH_SCALE
            route_widths.append(max(MIN_WIDTH, w))
        except Exception:
            route_widths.append(2.0)

    route_names.append(routes[i].name)

__all__ = [
    "ROUTES_PATH",
    "routes",
    "route_colors",
    "route_widths",
    "route_names",
]

if __name__ == "__main__":
    # Quick sanity check when run directly
    print(f"Loaded: {len(routes)} routes from {ROUTES_PATH}")
    if route_names:
        print("First names:", route_names[:3])
    if routes:
        print("First route starts at (json_start):", getattr(routes[0], "json_start", None))
