import agentpy
from enum import Enum
import numpy as np
from traffic_light import TLConnection, TrafficLightState
from typing import Any, Dict, Set
import random
import tl_regressors  # Use the dedicated module for traffic lights

class TLCtrlAction(Enum):
    SET_RED = 0
    SET_YELLOW = 1
    SET_GREEN = 2

class TLCtrlMode(Enum):
    FIXED = 1
    QLEARNING = 2

class TrafficLightController(agentpy.Agent):
    mode: TLCtrlMode
    tlcs: list[TLConnection] = []

    # Q-learning parameters
    q_models: Dict[TLCtrlAction, Any] = {}
    _trained_actions_idx: Set[int] = set()
    replay: Any = None

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 1.0
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.05

    warmup: int = 200
    train_every: int = 10

    state_duration: Dict[TLConnection, int] = {}

    def setup(self, tlcs: list[TLConnection], mode: TLCtrlMode = TLCtrlMode.FIXED): # type: ignore
        self.mode = mode
        self.tlcs = tlcs

        # Initialize Q-learning components if in QLEARNING mode
        if self.mode == TLCtrlMode.QLEARNING:
            # Get policy directory from model if available
            policy_dir = getattr(self.model, "policy_dir", "checkpoints")
            autosave_interval = int(getattr(self.model, "autosave_interval", 0))
            training = bool(getattr(self.model, "training_enabled", True))

            if training:
                tl_regressors.bootstrap(dirpath=policy_dir, 
                                      num_actions=len(TLCtrlAction),
                                      autosave_interval=autosave_interval)
            else:
                tl_regressors.load(policy_dir)

            # Wire shared objects
            self.q_models = {a: tl_regressors.models()[a.value] for a in TLCtrlAction}
            self._trained_actions_idx = tl_regressors.trained_actions()
            self.replay = tl_regressors.replay()

            # Exploration settings based on training mode
            self.exploration_rate = 1.0 if training else 0.0
            self.warmup = 200 if training else 0
            self.train_every = 10 if training else 0

            # Initialize last states and actions
            self.last_states = {tlc: None for tlc in self.tlcs}
            self.last_actions = {tlc: None for tlc in self.tlcs}

        self.state_duration = {tlc: 0 for tlc in tlcs}

    def step(self):
        # Update duration counters
        if not hasattr(self, 'last_tl_states'):
            self.last_tl_states = {tlc: tlc.traffic_light.state for tlc in self.tlcs}
        for tlc in self.tlcs:
            if tlc.traffic_light.state == self.last_tl_states[tlc]:
                self.state_duration[tlc] += 1
            else:
                self.state_duration[tlc] = 1
            self.last_tl_states[tlc] = tlc.traffic_light.state

        if self.mode == TLCtrlMode.FIXED:
            self._fixed_cycle()
        elif self.mode == TLCtrlMode.QLEARNING:
            self._qlearning_control()
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

    def _qlearning_control(self):
        """Control traffic lights using Q-learning"""
        training = bool(getattr(self.model, "training_enabled", True))
        
        for tlc in self.tlcs:
            # Get current state
            state = self._get_tl_state(tlc)
            
            # Choose action
            action = self._choose_action(state)
            
            # Apply action to traffic light
            self._apply_action(tlc, action)
            
            if not training or self.last_states[tlc] is None:
                # Store state for next step
                self.last_states[tlc] = state
                self.last_actions[tlc] = action
                continue
            
            # Calculate reward for previous action
            reward = self._calculate_reward(tlc)
            
            # Store transition
            self._remember(self.last_states[tlc], 
                          self.last_actions[tlc], 
                          reward, 
                          state, 
                          False)  # not terminal
            
            # Store current state for next step
            self.last_states[tlc] = state
            self.last_actions[tlc] = action
        
        # Periodically train from replay buffer
        if training and len(self.replay) >= self.warmup and self.train_every and (self.model.t % self.train_every == 0):
            self._train_from_replay(sample_size=256)
        
        # Decay exploration rate
        if training:
            self.exploration_rate = max(self.min_exploration_rate, 
                                      self.exploration_rate * self.exploration_decay)
    
    def _get_tl_state(self, tlc: TLConnection) -> np.ndarray:
        """
        Get state representation for a traffic light.
        Features include:
        - Current light state (one-hot)
        - Number of cars approaching the light
        - Average speed of approaching cars
        - Number of cars waiting at the light
        """
        # One-hot encoding of current state
        is_red = np.float32(1.0 if tlc.traffic_light.state == TrafficLightState.RED else 0.0)
        is_yellow = np.float32(1.0 if tlc.traffic_light.state == TrafficLightState.YELLOW else 0.0)
        is_green = np.float32(1.0 if tlc.traffic_light.state == TrafficLightState.GREEN else 0.0)
        
        # Get cars on this route
        route_cars = [car for car in self.model.active_cars if car.route == tlc.route]
        
        # Cars approaching the light (within 100 units)
        approaching_cars = [car for car in route_cars 
                            if car.s < tlc.s and tlc.s - car.s < 100.0]
        num_approaching = np.float32(len(approaching_cars))
        
        # Cars waiting at light (within 10 units and speed < 0.5)
        waiting_cars = [car for car in approaching_cars 
                       if tlc.s - car.s < 10.0 and car.ds < 0.5]
        num_waiting = np.float32(len(waiting_cars))
        
        # Average speed of approaching cars
        avg_speed = np.float32(0.0)
        if approaching_cars:
            avg_speed = np.float32(sum(car.ds for car in approaching_cars) / len(approaching_cars))
            
        # Normalize average speed
        avg_speed_norm = np.float32(avg_speed / 5.0)  # Assuming max speed is 5.0
        
        # Create feature vector
        features = [
            is_red, is_yellow, is_green,
            num_approaching / 10.0,  # Normalize by assuming max 10 cars
            num_waiting / 5.0,       # Normalize by assuming max 5 cars
            avg_speed_norm
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _choose_action(self, state: np.ndarray) -> TLCtrlAction:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            return random.choice(list(TLCtrlAction))
        
        q_values = []
        for a in TLCtrlAction:
            q_values.append(self._predict_q(state, a))
        
        # Add small noise for tie-breaking
        q_array = np.array(q_values, dtype=np.float32)
        q_array = q_array + 1e-6 * np.random.randn(*q_array.shape)
        best_idx = int(np.argmax(q_array))
        
        return list(TLCtrlAction)[best_idx]
    
    def _apply_action(self, tlc: TLConnection, action: TLCtrlAction) -> None:
        """Apply the selected action to the traffic light"""
        if action == TLCtrlAction.SET_RED:
            tlc.traffic_light.set(TrafficLightState.RED)
        elif action == TLCtrlAction.SET_YELLOW:
            tlc.traffic_light.set(TrafficLightState.YELLOW)
        elif action == TLCtrlAction.SET_GREEN:
            tlc.traffic_light.set(TrafficLightState.GREEN)
            
    def _calculate_reward(self, tlc: TLConnection) -> np.float32:
        """
        Calculate reward based on traffic flow efficiency
        - Positive reward for cars passing through green light
        - Negative reward for cars waiting at red light
        - Negative reward for empty green lights (wasted capacity)
        """
        route_cars = [car for car in self.model.active_cars if car.route == tlc.route]
        
        # Cars in proximity to the light
        nearby_cars = [car for car in route_cars 
                      if abs(car.s - tlc.s) < 20.0]
        
        # Cars approaching the light
        approaching_cars = [car for car in route_cars 
                           if car.s < tlc.s and tlc.s - car.s < 50.0]
        
        # Cars that recently passed the light
        passed_cars = [car for car in route_cars 
                      if car.s > tlc.s and car.s - tlc.s < 10.0 and car.ds > 1.0]
        
        reward = 0.0

        # Penalize changing state too quickly
        min_duration = 10
        if self.state_duration[tlc] < min_duration:
            reward -= 2.0  # Increase penalty as needed

        # Reward for current state
        if tlc.traffic_light.state == TrafficLightState.GREEN:
            # Reward for cars passing through green
            reward += len(passed_cars) * 2.0
            
            # Penalize empty green lights
            if len(approaching_cars) == 0:
                reward -= 0.5
        
        elif tlc.traffic_light.state == TrafficLightState.RED:
            # Penalize cars waiting at red light
            waiting_cars = [car for car in approaching_cars if car.ds < 0.5]
            reward -= len(waiting_cars) * 0.5
            
            # But reward if there are no cars to pass
            if len(nearby_cars) == 0:
                reward += 0.2
        
        return np.float32(reward)
    
    def _remember(self, s: np.ndarray, a: TLCtrlAction, r: np.float32, 
                 s_next: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer"""
        self.replay.append((
            s.astype(np.float32), 
            int(a.value), 
            float(r), 
            s_next.astype(np.float32), 
            bool(done)
        ))
    
    def _predict_q(self, state: np.ndarray, action: TLCtrlAction) -> np.float32:
        """Predict Q-value for state-action pair"""
        model = self.q_models.get(action)
        if model is None or action.value not in self._trained_actions_idx:
            return np.float32(0.0)
        
        try:
            pred = model.predict(state.reshape(1, -1))
            return np.float32(np.asarray(pred).item())
        except Exception as e:
            # Be tolerant during early training
            print(f"TL prediction error for action {action}: {e}")
            return np.float32(0.0)
    
    def _train_from_replay(self, sample_size: int | None = 512) -> None:
        """Train Q-function approximator from experiences"""
        data = list(self.replay)
        if not data:
            return
            
        if sample_size and len(data) > sample_size:
            data = random.sample(data, sample_size)
        
        # Get next states
        S2 = np.vstack([t[3] for t in data]).astype(np.float32)
        
        # Batch predict Q_next for each action
        q_next_mat = []
        for a in TLCtrlAction:
            if a.value in self._trained_actions_idx:
                q_next_mat.append(self.q_models[a].predict(S2))
            else:
                q_next_mat.append(np.zeros((S2.shape[0],), dtype=np.float32))
                
        q_next_max = np.max(np.vstack(q_next_mat), axis=0)
        
        # Prepare training data for each action's model
        gamma = self.discount_factor
        by_X = {a.value: [] for a in TLCtrlAction}
        by_y = {a.value: [] for a in TLCtrlAction}
        
        for i, (s, a_idx, r, s2, done) in enumerate(data):
            target = r if done else r + gamma * float(q_next_max[i])
            by_X[a_idx].append(s)
            by_y[a_idx].append(target)
        
        # Train each action's model
        for a in TLCtrlAction:
            X_list = by_X[a.value]
            if not X_list:
                continue
                
            X = np.vstack(X_list).astype(np.float32)
            y = np.asarray(by_y[a.value], dtype=np.float32)
            
            model = self.q_models[a]
            model.fit(X, y)
            self._trained_actions_idx.add(a.value)