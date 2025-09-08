import enum
from route import Route
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib import patches, transforms
from typing import Optional

Z_TRAFFIC_LIGHTS = 30   # above routes (~1) and cars (~2)


class TrafficLightState(enum.Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class TrafficLight:
    """
    A Traffic Light that lives in the Model and controls the flow of traffic.
    """
    _timer: float = 0.0
    _dur = {TrafficLightState.RED: 60, TrafficLightState.GREEN: 60, TrafficLightState.YELLOW: 12}

    def step(self, dt: float = 1.0) -> None:
        self._timer += dt
        if self._timer >= self._dur[self.state]:
            self._timer = 0.0
            self.state = {
                TrafficLightState.RED:    TrafficLightState.GREEN,
                TrafficLightState.GREEN:  TrafficLightState.YELLOW,
                TrafficLightState.YELLOW: TrafficLightState.RED,
            }[self.state]

        # keep visuals in sync
        if hasattr(self, "patch") and self.patch is not None:
            self.patch.set_facecolor(self.state.value.lower())
            self.patch.set_edgecolor("black")
            self.patch.set_linewidth(2.0)
            self.patch.set_zorder(Z_TRAFFIC_LIGHTS)


    state: TrafficLightState
    patch: Optional[Rectangle] = None

    # The universal location where the traffic light will be rendered for all routes
    x: float
    y: float
    rotation: float = 0.0

    def __init__(self, x: float, y: float, rotation: float = 0.0):
        self.state = TrafficLightState.RED
        self.x = x
        self.y = y
        self.rotation = rotation

    def plot(self, ax):
        w, h = 7, 35
        x0, y0 = self.x - w/2, self.y - h/2

        if not hasattr(self, "patch") or self.patch is None:
            self.patch = patches.Rectangle(
                (x0, y0), w, h,
                facecolor=self.state.value.lower(),   # ← use facecolor (no warning)
                edgecolor="black",                    # ← black outline
                linewidth=2.0,
                zorder=Z_TRAFFIC_LIGHTS               # ← draw above cars/routes
            )
            ax.add_patch(self.patch)
        else:
            self.patch.set_xy((x0, y0))
            self.patch.set_zorder(Z_TRAFFIC_LIGHTS)

        self.patch.set_transform(
            transforms.Affine2D().rotate_deg_around(self.x, self.y, self.rotation)
            + ax.transData
        )

class TLConnection:
    """
    Assigns a traffic light to a Route and sets the range where the car should stop.
    """
    route: Route
    s: float # The distance in the route at which the car will stop.
    traffic_light: TrafficLight

    def __init__(self, route: Route, traffic_light: TrafficLight, s: float):
        self.route = route
        self.s = s
        self.traffic_light = traffic_light