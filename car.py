import agentpy as ap

class Car(ap.Agent):
    """Simple car that follows a Route by advancing its arc-length s each step.

    Parameters (from model.p):
    - dt: time step in seconds
    - v_init: initial speed (world units/s)
    - v_max: hard speed cap (world units/s)
    - min_gap: desired minimum front gap if following a leader on same route
    - acc: simple acceleration when below desired speed
    - dec: simple deceleration when too close to leader or at route end
    """

    def setup(self, route: Route, s0: float = 0.0, v0: Optional[float] = None, color: str = "k"):
        self.route = route
        self.s = float(s0)
        self.v = float(v0 if v0 is not None else self.p.v_init)
        self.color = color  # used for plotting
        self.leader: Optional[Car] = None  # car ahead on same route, if any

    # Position helpers
    @property
    def xy(self) -> Tuple[float, float]:
        return self.route.pos_at(self.s)

    def step(self):
        dt = float(self.p.dt)
        v_max = float(min(self.p.v_max, self.route.speed_limit or self.p.v_max))
        a = float(self.p.acc)
        d = float(self.p.dec)
        min_gap = float(self.p.min_gap)

        # Very simple car-following: if leader exists on same route, keep gap
        target_v = v_max
        if self.leader is not None and self.leader.route is self.route:
            gap = max(0.0, self.leader.s - self.s)
            # If too close, reduce target speed proportional to gap
            if gap < min_gap:
                target_v = min(target_v, max(0.0, (gap / max(min_gap, 1e-3)) * v_max))

        # Approach end of route
        dist_to_end = max(0.0, self.route.length - self.s)
        if dist_to_end < max(self.v * dt, 1e-6):
            # Brake when close to the end unless the model decides to transfer to next route
            target_v = 0.0

        # Accelerate or decelerate to target_v
        if self.v < target_v:
            self.v = min(target_v, self.v + a * dt)
        else:
            self.v = max(target_v, self.v - d * dt)

        # Advance along the route
        self.s = min(self.route.length, self.s + self.v * dt)

