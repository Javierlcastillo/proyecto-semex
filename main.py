from model import Renderer

model = Renderer({
    'dt': 1.0,
    'MAX_CARS': 30,        # global cap
    'HEADWAY_MEAN': 10.0,   # avg seconds between arrivals per route
    # optional tunables:
    # 'SPEED_MIN': 3.5, 'SPEED_MAX': 6.5,
    # 'MIN_GAP': 25.0, 'SPAWN_S': 0.0
})
model.run(steps=3000)
