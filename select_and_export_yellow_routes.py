# select_and_export_yellow_routes.py
import json
import numpy as np
import matplotlib.pyplot as plt

from defined_routes import routes   # uses datos/routes.json through your existing loader

def sample_route_points(route, n=200):
    """Return n evenly-spaced (x,y) points along a Route."""
    pts = route.sample_even(n)
    return [{"x": float(x), "y": float(y)} for x, y in pts]

def nearest_route_index(x, y, K=200):
    """Pick the route whose polyline comes closest to the clicked (x,y)."""
    p = np.array([x, y], dtype=float)
    best_i, best_d = -1, float("inf")
    for i, r in enumerate(routes):
        pts = r.sample_even(K)
        d = np.min(np.linalg.norm(pts - p, axis=1))
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d

def main():
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_aspect('equal', adjustable='box')
    # Show all routes (thin) so you can pick yellows
    colors = ["#d3d3d3"] * len(routes)         # grey baseline
    for i, r in enumerate(routes):
        pts = r.sample_even(400)
        ax.plot(pts[:,0], pts[:,1], color=colors[i], lw=2)

    ax.set_title("Click each YELLOW route once, then press Enter")
    ax.grid(True, alpha=0.25)

    # Collect clicks
    clicks = plt.ginput(n=-1, timeout=0)
    print(f"Clicks: {len(clicks)}")

    chosen = []
    for (x, y) in clicks:
        idx, dist = nearest_route_index(x, y)
        if idx not in chosen:
            chosen.append(idx)

    chosen.sort()
    print("Selected route indices (yellow):", chosen)

    # Export: Unity-friendly JSON
    out = {"routes": []}
    for k, idx in enumerate(chosen):
        r = routes[idx]
        out["routes"].append({
            "name": f"Y{k:02d}_idx{idx}",
            "pts": sample_route_points(r, n=250)   # adjust density if needed
        })

    with open("yellow_routes.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote yellow_routes.json with", len(out["routes"]), "routes.")

    # Optional: re-draw chosen in yellow
    for idx in chosen:
        pts = routes[idx].sample_even(400)
        ax.plot(pts[:,0], pts[:,1], color="gold", lw=3)
    plt.show()

if __name__ == "__main__":
    main()
