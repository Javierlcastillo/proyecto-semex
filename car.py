import agentpy as ap
from route import Route, Point
import numpy as np
from matplotlib import patches, transforms, axes
from typing import Optional, Any
from traffic_light import TLConnection
from sklearn.ensemble import ExtraTreesRegressor

class Car(ap.Agent):   
    
    route: Route
    tlconnections: list[TLConnection]

    position: Point
    ds: float

    maxRateOfAceleration: float
    width: float
    height: float
    is_colliding: bool

    patch: Optional[patches.Rectangle]

    @property
    def nextTLC(self):
        nearest: Optional[TLConnection] = None

        for tlc in self.tlconnections:
            if tlc.route == self.route and self.s < tlc.s and (nearest is None or tlc.s < nearest.s):
                nearest = tlc
        return nearest

    def setup(self, route: Route, tlconnections: list[TLConnection]): # pyright: ignore[reportIncompatibleMethodOverride]
        self.route = route
        self.tlconnections = tlconnections

        self.s = 0.0
        self.position = self.route.pos_at(self.s)

        self.maxRateOfAceleration = np.random.uniform(0.8, 1.2)

        self.ds = 1  # Amount of S that changes in a step

        self.width = 20
        self.height = 10
        self.is_colliding = False
        self.experience_buffer = []  # For fitted Q-learning

    def step(self, cars: list["Car"], q_regressor=None):
        # Q-learning experience collection
        prev_state = self.get_state(cars)
        # Action selection using fitted Q-function if regressor is provided
        if q_regressor is not None:
            actions = [0, 1]  # Define your action space
            inputs = [np.concatenate([prev_state, [a]]) for a in actions]
            q_values = q_regressor.predict(inputs)
            action = actions[np.argmax(q_values)]
        else:
            action = self.ds  # Default action if no regressor
        self.s += action
        self.position = self.route.pos_at(self.s)
        next_state = self.get_state(cars)
        # Update collision status before reward calculation
        self.update_collision(cars)
        reward = self.calc_reward(prev_state, action, next_state, cars)
        self.experience_buffer.append((prev_state, action, reward, next_state))
        #print("Car state:", next_state)

    def calc_reward(self, prev_state, action, next_state, cars: list["Car"]):
        # Basic reward function
        reward = 0.0
        # Negative reward for collision
        if self.is_colliding:
            reward -= 100.0
        # Negative reward for running a red light
        tlc = self.nextTLC
        if tlc and hasattr(tlc.traffic_light, 'state') and tlc.traffic_light.state == 'red':
            # If car is close to intersection and light is red
            if self.s >= tlc.s - self.width and self.s <= tlc.s + self.width:
                reward -= 50.0
        # Positive reward for reaching route end
        if hasattr(self.route, 'length') and self.s >= self.route.length:
            reward += 200.0
        # Small negative reward for time spent
        reward -= 1.0
        return reward
    

    def get_state(self, cars: list["Car"]) -> np.ndarray:
        # Position along route
        s = self.s
        # Speed
        ds = self.ds
        # Distance to car ahead (on same route, with higher s)
        cars_ahead = [car for car in cars if car.route == self.route and car.s > self.s]
        if cars_ahead:
            dist_to_ahead = min(car.s - self.s for car in cars_ahead)
        else:
            dist_to_ahead = float('inf')
        # Traffic light state (0=red, 1=green, 0.5=yellow, fallback to 1 if not found)
        tlc = self.nextTLC
        if tlc and hasattr(tlc.traffic_light, 'state'):
            tl_state = tlc.traffic_light.state
            if tl_state == 'red':
                tl_val = 0.0
            elif tl_state == 'green':
                tl_val = 1.0
            elif tl_state == 'yellow':
                tl_val = 0.5
            else:
                tl_val = 1.0
        else:
            tl_val = 1.0
        # Remaining distance to route end
        if hasattr(self.route, 'length'):
            rem_dist = self.route.length - self.s
        else:
            rem_dist = 0.0
        state = np.array([s, ds, dist_to_ahead, tl_val, rem_dist], dtype=np.float32)
        # Replace inf, -inf, nan with finite values
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        return state

    def plot(self, ax: axes.Axes):
        x, y = self.position
        w, h = self.width, self.height

        if not hasattr(self, "patch") or self.patch is None:
            self.patch = patches.Rectangle(
                (x - w/2, y - h/2), w, h,
                facecolor='black', edgecolor='white', linewidth=1, zorder=2
            )
            ax.add_patch(self.patch)  # Add patch to Axes
        else:
            self.patch.set_xy((x - w/2, y - h/2))

        angle = self.heading()
        self.patch.set_transform(
            transforms.Affine2D().rotate_around(x, y, angle) + ax.transData
        )        

    def accelerate(self, rate: float):
        if np.abs(rate) < np.abs(self.maxRateOfAceleration):
            self.ds *= rate

    # --- Collision helpers on the agent ---

    def heading(self) -> float:
        p0 = self.route.pos_at(self.s)
        p1 = self.route.pos_at(self.s + self.ds)
        return float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))  # radians

    def corners(self) -> np.ndarray[Any, Any]:
        # Oriented rectangle corners (counter-clockwise) in world coords
        x, y = self.position
        w, h = self.width, self.height
        a = self.heading()
        c, s = np.cos(a), np.sin(a)
        hw, hh = w/2.0, h/2.0
        local = np.array([
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh],
        ])
        R = np.array([[c, -s],[s, c]])
        return local @ R.T + np.array([x, y])

    @staticmethod
    def _sat_overlap(A: np.ndarray[Any, Any], B: np.ndarray[Any, Any]) -> bool:
        # A and B: (4,2) arrays of rectangle corners
        def axes_from(poly: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
            edges = np.roll(poly, -1, axis=0) - poly
            axes = []
            for e in edges[:2]:  # two unique edge directions for a rectangle
                n = np.array([-e[1], e[0]])
                ln = np.linalg.norm(n)
                if ln > 1e-9:
                    axes.append(n / ln) # type: ignore
            return axes # type: ignore

        for axis in axes_from(A) + axes_from(B):
            projA = A @ axis
            projB = B @ axis
            if projA.max() < projB.min() or projB.max() < projA.min():
                return False
        return True

    def collides_with(self, other: "Car") -> bool:
        if self is other:
            return False
        return self._sat_overlap(self.corners(), other.corners())

    def update_collision(self, cars: list["Car"]) -> None:
        self.is_colliding = any(self.collides_with(o) for o in cars if o is not self)
