import enum
from route import Route
from matplotlib.axes import Axes
from matplotlib.patches import Circle
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
    patch: Optional[Circle] = None

    # The universal location where the traffic light will be rendered for all routes
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.state = TrafficLightState.RED
        self.x = x
        self.y = y

    def plot(self, ax: Axes):
        if not self.patch:
            self.patch = Circle((self.x, self.y), 5, color=self.state.value.lower(), zorder=2)
            ax.add_patch(self.patch) 
        else:
            self.patch.set_color(self.state.value.lower())

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