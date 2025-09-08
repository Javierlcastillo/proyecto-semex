import json
from route import Route

routes: list[Route] = []

with open("datos/routes.json", "r", encoding="utf-8") as f:
#with open("yellow_routes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for points in data:
    r = Route.from_polyline(points, samples_per_segment=50)
    routes.append(r)
