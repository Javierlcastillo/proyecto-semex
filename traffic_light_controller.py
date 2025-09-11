import agentpy
from enum import Enum
from traffic_light import TLConnection, TrafficLightState

class TrafficLightControllerMode(Enum):
    FIXED = 1
    ADAPTIVE = 2

class TrafficLightController(agentpy.Agent):
    mode: TrafficLightControllerMode
    tlcs: list[TLConnection] = []

    def setup(self, tlcs: list[TLConnection], mode: TrafficLightControllerMode = TrafficLightControllerMode.FIXED): # type: ignore
        self.mode = mode
        self.tlcs = tlcs

    def step(self):
        if self.mode == TrafficLightControllerMode.FIXED:
            self._fixed_cycle()
        elif self.mode == TrafficLightControllerMode.ADAPTIVE:
            self._adaptive_control()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
    def _fixed_cycle(self):
        cycle_length = 180  # Total cycle length in steps
        green_duration = 60  # Duration of green light in steps
        yellow_duration = 60  # Duration of yellow light in steps
        for i, tlc in enumerate(self.tlcs):
            step_in_cycle = (self.model.t + (i * cycle_length / len(self.tlcs))) % cycle_length
            if step_in_cycle < green_duration:
                tlc.traffic_light.set(TrafficLightState.GREEN)
            elif step_in_cycle < green_duration + yellow_duration:
                tlc.traffic_light.set(TrafficLightState.YELLOW)
            else:
                tlc.traffic_light.set(TrafficLightState.RED)