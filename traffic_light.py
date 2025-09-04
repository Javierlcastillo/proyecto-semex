import enum
from route import Route
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle

class TrafficLightState(enum.Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class TrafficLight:
    """
    A Traffic Light that lives in the Model and controls the flow of traffic.
    """

    state: TrafficLightState

    # The universal location where the traffic light will be rendered for all routes
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.state = TrafficLightState.RED
        self.x = x
        self.y = y

    def plot(self, ax: Axes):
        circle = Circle((self.x, self.y), 5, color=self.state.value)
        ax.add_patch(circle)

class TLConnection:
    """
    Assigns a traffic light to a Route and sets the range where the car should stop.
    """
    route: Route
    s: float # The distance in the route at which the car will stop.
    traffic_light: TrafficLight

    def __init__(self, route: Route, s: float, traffic_light: TrafficLight):
        self.route = route
        self.s = s
        self.traffic_light = traffic_light