import numpy as np
import pickle

class QLearner:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        q_values = self.get_q_values(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        action_index = self.actions.index(action)
        td_target = reward + self.gamma * np.max(next_q_values)
        td_error = td_target - q_values[action_index]
        q_values[action_index] += self.alpha * td_error

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

# Nota: El ciclo de entrenamiento debe llamar a choose_action(state) para seleccionar la acción,
# aplicar esta acción en el modelo (model.py y car.py), y luego llamar a update(state, action, reward, next_state).
# Esto permite aprender los valores Q basados en la experiencia.
    # Wrapper para facilitar la integración con el modelo y obtener el estado del auto
    def get_state_from_model(self, model, car):
        return model.get_state(car)

    def apply_action_to_model(self, model, car, action):
        # Esto se usa para aplicar la acción seleccionada al auto en el modelo.
        if hasattr(model, 'apply_action'):
            model.apply_action(car, action)