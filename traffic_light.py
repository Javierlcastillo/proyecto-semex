
    
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
    id: Optional[str]  # <- declarado para el type checker
    _timer: float = 0.0
    _dur = {TrafficLightState.RED: 60, TrafficLightState.GREEN: 60, TrafficLightState.YELLOW: 12}

    def step(self, dt: float = 1.0) -> None:
        # Solo actualiza la visualización, no cambia el estado automáticamente
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
        self.state = TrafficLightState.GREEN
        self.x = x
        self.y = y
        self.width_px = 7.0         # visual width of the head in Python coords
        self.arm_len  = 35.0        # how far the head extends from the pivot
        self.rotation = rotation
        self.id = None      # <- valor inicial

    def plot(self, ax):
        w = self.width_px
        h = self.arm_len

        # Anchor: pivot (self.x, self.y) = bottom-center of the head
        x0 = self.x - w/2
        y0 = self.y

        if not hasattr(self, "patch") or self.patch is None:
            self.patch = patches.Rectangle(
                (x0, y0), w, h,
                facecolor=self.state.value.lower(),
                edgecolor="black",
                linewidth=2.0,
                zorder=Z_TRAFFIC_LIGHTS,
            )
            ax.add_patch(self.patch)
        else:
            self.patch.set_xy((x0, y0))
            self.patch.set_zorder(Z_TRAFFIC_LIGHTS)

        # Rotate around the mast pivot
        self.patch.set_transform(
            transforms.Affine2D().rotate_deg_around(self.x, self.y, self.rotation)
            + ax.transData
        )
    
    def heuristic_control(self, queue, autos_en_rotonda, pairs, tl_by_id, params, cars=None, tlconnections=None, can_turn_green=True):
        # Lógica robusta: los pares nunca pueden estar en verde al mismo tiempo
        
        # Debug: imprimir cantidad de carros detectados adelante
        
        # --- Restricción de pares básica ---
        
        """
        Controla el estado del semáforo usando heurística, según el contexto recibido.
        params: dict con MIN_RED, MAX_RED, MIN_GREEN, MAX_GREEN, QUEUE_THRESHOLD, ROTONDA_THRESHOLD
        """
        # Inicializa contadores si no existen
        if not hasattr(self, "_timer_heuristic"):
            self._timer_heuristic = 0
            self._last_state = self.state

        self._timer_heuristic += 1
        tl_id = getattr(self, "id", None)

        MIN_RED = params.get("MIN_RED", 30)
        MAX_RED = params.get("MAX_RED", 120)
        MIN_GREEN = params.get("MIN_GREEN", 60)
        MAX_GREEN = params.get("MAX_GREEN", 120)
        QUEUE_THRESHOLD = params.get("QUEUE_THRESHOLD", 4)
        ROTONDA_THRESHOLD = params.get("ROTONDA_THRESHOLD", 10)

        # Si hay demasiados autos en la rotonda, no deja pasar
        #if autos_en_rotonda >= ROTONDA_THRESHOLD:
        #    if self.state != TrafficLightState.RED:
        #        self.state = TrafficLightState.RED
        #        self._timer_heuristic = 0
        #    return

        # --- Sin restricción de pares ---
        can_turn_green = True
        #for id1, id2 in pairs:
        #    if self.id == id1:
        #        other = tl_by_id.get(id2)
        #        if other and other.state == TrafficLightState.GREEN:
        #            return
        #            can_turn_green = False
        #    elif self.id == id2:
        #        other = tl_by_id.get(id1)
        #        if other and other.state == TrafficLightState.GREEN:
        #            return
        #            can_turn_green = False

        # Verifica si hay demasiados carros después del semáforo en la ruta
    # Lógica de detección de carros después desactivada
        too_many_after = False

    # can_turn_green ahora lo decide el modelo y se pasa como argumento

        if self.state == TrafficLightState.RED:
            if queue >= QUEUE_THRESHOLD and self._timer_heuristic >= MIN_RED and can_turn_green:
                self.state = TrafficLightState.GREEN
                self._timer_heuristic = 0
            elif self._timer_heuristic >= MAX_RED and can_turn_green:
                self.state = TrafficLightState.GREEN
                self._timer_heuristic = 0

        # Cambia a rojo solo si cumplió el mínimo en verde, independientemente de la cola
        elif self.state == TrafficLightState.GREEN:
            if self._timer_heuristic >= MIN_GREEN:
                if queue < QUEUE_THRESHOLD or self._timer_heuristic >= MAX_GREEN:
                    self.state = TrafficLightState.RED
                    self._timer_heuristic = 0

        # Amarillo: transición automática
        elif self.state == TrafficLightState.YELLOW:
            if self._timer_heuristic >= 12:
                self.state = TrafficLightState.RED
                self._timer_heuristic = 0


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

    