import enum
from route import Route
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from typing import Optional

class TrafficLightState(enum.Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class TrafficLight:
    """
    A Traffic Light that lives in the Model and controls the flow of traffic.
    """
    state: TrafficLightState
    patch: Optional[Rectangle] = None

    # A reference to the model's Axes
    ax: Optional[Axes] = None

    # The universal location where the traffic light will be rendered for all routes
    x: float
    y: float
    rotation: float = 0.0

    def __init__(self, x: float, y: float, ax: Optional[Axes] = None, rotation: float = 0.0):
        self.state = TrafficLightState.RED
        self.x = x
        self.y = y
        self.rotation = rotation
        self.ax = ax
    @property
    def rotation_deg(self) -> float:
        return float(self.rotation)

    @rotation_deg.setter

    
    def rotation_deg(self, val: float) -> None:
        self.rotation = float(val)
    def plot(self):
        if self.ax is None:
            return
        
        if not self.patch:
            self.patch = Rectangle((self.x, self.y), 7, 35, color=self.state.value.lower(), edgecolor='black', zorder=2)

            self.patch.set_transform(
                Affine2D().rotate_deg_around(self.x, self.y, self.rotation) + self.ax.transData
            )
            self.ax.add_patch(self.patch)
        else:
            self.patch.set_color(self.state.value.lower())

    def set(self, state: TrafficLightState):
        self.state = state
        self.plot()

class TLConnection:
    """
    Assigns a traffic light to a Route and sets the range where the car should stop.
    """
    route: Route
    traffic_light: TrafficLight
    s: float # The distance in the route at which the car will stop.

    def __init__(self, route: Route, traffic_light: TrafficLight, s: float):
        self.route = route
        self.s = s
        self.traffic_light = traffic_light