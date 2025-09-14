from __future__ import annotations
import agentpy as ap
import enum, numpy as np
import car_regressors
from route import Route, Point
from collections import defaultdict
from matplotlib import patches, transforms, axes
from typing import Optional, Any, TYPE_CHECKING
from traffic_light import TLConnection, TrafficLightState
import random

if TYPE_CHECKING:
    from model import Model

class CarAction(enum.Enum):
    Accelerate = 0
    Decelerate = 1
    Maintain = 2

class Car(ap.Agent):
    model: Model
    route: Route
    tlconnections: list[TLConnection]

    ds: float

    maxRateOfAceleration: float
    width: float
    height: float

    spawn_step: int

    patch: Optional[patches.Polygon] = None

    # Q-Learning attributes
    q_table: defaultdict[tuple, np.ndarray]
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    exploration_decay: float
    min_exploration_rate: float

    q_models: dict[CarAction, Any]  
    _trained_actions_idx: set[int]
    replay: Any 

    warmup: int
    train_every: int

    def setup(self, route: Route, tlconnections: list[TLConnection]):  # pyright: ignore[reportIncompatibleMethodOverride]
        # agentpy sets self.model
        self.route = route
        self.tlconnections = tlconnections

        # Physics
        self.s = 0.0
        self.ds = 1.0
        self.maxRateOfAceleration = np.random.uniform(0.8, 1.2)
        self.width, self.height = 20.0, 10.0
        self.patch = None

        self.spawn_step = -1

        # Q-learning params
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

        # Shared policy: load or init+autosave based on mode
        policy_dir = getattr(self.model, "policy_dir", "checkpoints")
        autosave_interval = int(getattr(self.model, "autosave_interval", 0))
        training = bool(getattr(self.model, "training_enabled", True))

        if training:
            car_regressors.bootstrap(dirpath=policy_dir, num_actions=len(CarAction), autosave_interval=autosave_interval)
        else:
            # Load-only, no autosave thread
            car_regressors.load(policy_dir)

        # Wire shared objects
        self.q_models = { a: car_regressors.models()[a.value] for a in CarAction }
        self._trained_actions_idx = car_regressors.trained_actions()
        self.replay = car_regressors.replay()

        # Exploration: off in eval
        self.exploration_rate = 1.0 if training else 0.0

        self.warmup = 200 if training else 0
        self.train_every = 10 if training else 0

    @property
    def pos(self) -> Point:
        return self.route.pos_at(self.s)

    @property
    def nextTlc(self):
        nearest: Optional[TLConnection] = None

        for tlc in self.tlconnections:
            if tlc.route == self.route and self.s < tlc.s and (nearest is None or tlc.s < nearest.s):
                nearest = tlc
        return nearest

    def step(self):
        state = self.get_state()
        action = self.choose_action(state)
        self.perform_action(action)
        next_state = self.get_state()

        training = bool(getattr(self.model, "training_enabled", True))
        if not training:
            return  # eval-only: no memory, no fitting, no epsilon decay

        # Reward + terminal
        killed = self.is_colliding 
        done = (hasattr(self.route, "length") and self.s >= float(self.route.length))
        reward = self.calculate_reward(state, action)

        # Store + periodic train
        self.remember(state, action, reward, next_state, done)

        if len(self.replay) >= self.warmup and self.train_every and (self.model.t % self.train_every == 0):
            self.train_from_replay(sample_size=256)

        if done:
            self.reset_episode()
            self.model.finished_cars += 1
            print(f"Car finished route at step {self.model.t}. Total finished this step: {self.model.finished_cars}")
        if killed:
            self.reset_episode()
            print(f"Car collided at step {self.model.t}")

        # Epsilon decay
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def reset_episode(self) -> None:
        """Respawn the car to continue training after a terminal state."""
        # Clamp inside route, respawn at start (or randomize within first segment)
        self.s = 0.0
        self.ds = max(0.0, np.random.uniform(0.5, 2.0))

    def plot(self, ax: axes.Axes):
        corners = self.corners
        if self.patch is None:
            self.patch = patches.Polygon(
                corners,
                closed=True,
                facecolor= 'tab:red' if self.is_colliding else 'tab:blue',
                alpha=0.9,
                zorder=2
            )
            ax.add_patch(self.patch)
        else:
            self.patch.set_xy(corners)

    def remove(self):
        if self.patch is not None:
            self.patch.remove()
            self.patch = None

    @property
    def heading_in_radians(self) -> float:
        """
        Returns the heading angle in radians based on current and next position.
        """
        p0 = self.route.pos_at(self.s)
        p1 = self.route.pos_at(self.s + self.ds)
        return float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

    @property
    def corners(self) -> np.ndarray[Any, Any]:
        """
        Returns the corner points of the car's rectangle in world coordinates.
        """
        x, y = self.pos
        w, h = self.width, self.height
        a = self.heading_in_radians
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
        """
        Check for overlap between two oriented rectangles A and B using the Separating Axis Theorem.
        A and B: (4,2) arrays of rectangle corners
        """
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

    @property
    def is_colliding(self) -> bool:
        # Fast path: use model-level cache if available
        flags = getattr(self.model, "collision_flags", None)
        if isinstance(flags, dict):
            hit = flags.get(self, None)
            if hit is not None:
                return bool(hit)
        # Fallback (only if cache not ready)
        # Use active_cars instead of cars
        for other in self.model.active_cars:
            if other is self:
                continue
            if self._sat_overlap(self.corners, other.corners):
                return True
        return False
    
    @property 
    def time_to_collision(self) -> np.float32:
        """
        Estimate time to collision (TTC) with nearest car.
        """
        x, y = self.pos
        theta = self.heading_in_radians
        c, s = np.cos(theta), np.sin(theta)
        v_ego = np.array([self.ds * c, self.ds * s], dtype=np.float32)

        min_ttc = np.float32(100.0)
        # Use active_cars instead of cars
        for other in self.model.active_cars:
            if other is self:
                continue
            ox, oy = other.pos
            dx, dy = ox - x, oy - y
            dist = np.float32(np.hypot(dx, dy))

            if dist < 1e-6 or dist > 100.0:
                continue
            oc, os = np.cos(other.heading_in_radians), np.sin(other.heading_in_radians)
            v_other = np.array([other.ds * oc, other.ds * os], dtype=np.float32)
            v_rel = v_other - v_ego
            closing_speed = np.float32(- (dx * v_rel[0] + dy * v_rel[1]) / dist)

            if closing_speed > 0:
                ttc = dist / closing_speed
                if ttc < min_ttc:
                    min_ttc = ttc

        return np.float32(min_ttc)

    """ Q-learning section """

    def get_state(self) -> np.ndarray:
        """
        Minimal fitted-Q state:
        [speed_norm, cos(theta), sin(theta), tl_dist_norm, tl_red, tl_yellow, tl_green, ttc_norm]
        """

        # Constants for scaling
        V_MAX = 5.0  # matches your perform_action cap
        TTC_CAP = 10.0  # steps; cap/normalize TTC

        # Ego features
        theta = self.heading_in_radians
        c, s = np.cos(theta), np.sin(theta)

        speed_norm = np.float32(self.ds / V_MAX)
        heading_cos = np.float32(c)
        heading_sin = np.float32(s)

        # Next traffic light along own route
        radius = 150.0
        tlc = self.nextTlc
        if tlc is not None and tlc.s >= self.s:
            tl_dist = tlc.s - self.s  # along-route distance
            # Normalize by a reasonable scale (use radius as generic length scale)
            tl_dist_norm = np.float32(np.clip(tl_dist / radius, 0.0, 1.0))
            state_val = tlc.traffic_light.state
            tl_red = np.float32(1.0 if state_val == TrafficLightState.RED else 0.0)
            tl_yellow = np.float32(1.0 if state_val == TrafficLightState.YELLOW else 0.0)
            tl_green = np.float32(1.0 if state_val == TrafficLightState.GREEN else 0.0)
        else:
            # No light ahead on this route
            tl_dist_norm = np.float32(1.0)
            tl_red = tl_yellow = tl_green = np.float32(0.0)

        ttc = min(float(self.time_to_collision), TTC_CAP)
        ttc_norm = np.float32(ttc / TTC_CAP)
        
        features = [
            speed_norm, heading_cos, heading_sin,
            tl_dist_norm, tl_red, tl_yellow, tl_green, 
            ttc_norm
        ]

        return np.array(features, dtype=np.float32)
    
    def perform_action(self, action: CarAction) -> None:
        """Apply action to speed (ds) and update position (s)."""
        V_MAX = 5.0
        accel = self.maxRateOfAceleration

        if action == CarAction.Accelerate:
            self.ds = min(self.ds + accel, V_MAX)
        elif action == CarAction.Decelerate:
            self.ds = max(self.ds - accel, 0.0)
        elif action == CarAction.Maintain:
            pass  # keep current speed

        # Advance along parametric distance
        self.s += self.ds

    def calculate_reward(self, state: np.ndarray, action: CarAction) -> np.float32:
        """
        Reward function based on speed, collisions, and traffic light compliance.
        """
        reward = 0.0

        # Progress along route
        reward += 0.1 * np.float32(self.ds)

        # Collision penalty
        if self.is_colliding:
            reward -= 5.0

        # Traffic light compliance
        tlc = self.nextTlc
        if tlc is not None and tlc.s >= self.s and tlc.traffic_light.state == TrafficLightState.RED:
            dist = tlc.s - self.s
            if dist < 10.0 and self.ds > 0.5:
                reward -= 1.0 # Approaching red too fast
            if dist < 1.0 and self.ds > 0.0:
                reward -= 2.0 # Running red light

        return np.float32(reward)
    
    def remember(self, s: np.ndarray, a: CarAction, r: np.float32, s_next: np.ndarray, done: bool) -> None:
        """Push transition into the shared replay buffer."""
        self.replay.append((s.astype(np.float32), int(a.value), float(r), s_next.astype(np.float32), bool(done)))

    def _model_is_ready(self, m):
        # Consider the model fitted if sklearn marked input shape or it has trees
        if hasattr(m, "n_features_in_"):
            return True
        est = getattr(m, "estimators_", None)
        return isinstance(est, (list, tuple)) and len(est) > 0


    def train_from_replay(self, sample_size: int | None = 512) -> None:
        # Uses replay buffer to fit per-action regressors; safe when models aren't fitted yet.
        data = list(self.replay)
        if not data:
            return

        if sample_size is not None and len(data) > sample_size:
            data = random.sample(data, sample_size)

        # transitions are (s, a_idx, r, s2, done)
        S2 = np.vstack([t[3] for t in data]).astype(np.float32)

        # ----- predict Q_next for each action (guard for unfitted models) -----
        q_next_rows = []
        for a in CarAction:
            m = self.q_models[a]
            ready = (
                hasattr(m, "n_features_in_")
                or (hasattr(m, "estimators_") and isinstance(m.estimators_, (list, tuple)) and len(m.estimators_) > 0)
            )
            if ready:
                try:
                    preds = m.predict(S2)
                except Exception as e:
                    print(f"[WARN] predict failed for {a}: {e}")
                    preds = np.zeros((S2.shape[0],), dtype=np.float32)
            else:
                preds = np.zeros((S2.shape[0],), dtype=np.float32)
            q_next_rows.append(preds.astype(np.float32))

        q_next_max = np.max(np.vstack(q_next_rows), axis=0)

        # ----- build targets per action -----
        gamma = getattr(self, "discount_factor", 0.99)
        by_X = {a.value: [] for a in CarAction}
        by_y = {a.value: [] for a in CarAction}

        for i, (s, a_idx, r, s2, done) in enumerate(data):
            target = r if done else r + gamma * float(q_next_max[i])
            by_X[a_idx].append(s)
            by_y[a_idx].append(target)

        # ----- fit each action model on its own minibatch -----
        for a in CarAction:
            X_list = by_X[a.value]
            if not X_list:
                continue

            X = np.vstack(X_list).astype(np.float32)
            y = np.asarray(by_y[a.value], dtype=np.float32)

            m = self.q_models[a]
            try:
                m.fit(X, y)
                # Mark as trained only if sklearn actually fitted it
                if hasattr(m, "n_features_in_") or (hasattr(m, "estimators_") and len(getattr(m, "estimators_", [])) > 0):
                    self._trained_actions_idx.add(a.value)
            except Exception as e:
                print(f"[WARN] fit failed for {a}: {e}")


    def choose_action(self, state: np.ndarray) -> CarAction:
        if np.random.random() < self.exploration_rate:
            return random.choice(list(CarAction))

        q_values: list[np.float32] = []
        for a in CarAction:
            q_values.append(self.predict_q(state, a))

        # Tie-breaking with small noise to avoid bias
        q_array = np.asarray(q_values, dtype=np.float32)
        q_array = q_array + 1e-6 * np.random.randn(*q_array.shape)
        best_idx = int(np.argmax(q_array))
        
        return list(CarAction)[best_idx]
    
    def predict_q(self, state: np.ndarray, action: CarAction) -> np.float32:
        """
        Predict Q(s, a) using the fitted regressor for that action.
        Returns 0.0 if the model is not yet trained for that action.
        """
        model = self.q_models.get(action)
        if model is None or action.value not in self._trained_actions_idx:
            return np.float32(0.0)
        try:
            pred = model.predict(state.reshape(1, -1))
            return np.float32(np.asarray(pred).item())
        except Exception as e:
            # Be tolerant during early training
            print(f"Prediction error for action {action}: {e}")
            return np.float32(0.0)
