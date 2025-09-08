from model import Renderer
from sklearn.ensemble import ExtraTreesRegressor

model = Renderer({ 'steps': 2000, 'nCars' : 50})
model.run()

for idx, car in enumerate(model.cars):
    print(f"Car {idx} buffer length: {len(car.experience_buffer)}")
    if car.experience_buffer:
        print("Sample experience:", car.experience_buffer[0])

q_regressor = ExtraTreesRegressor(n_estimators=10, min_samples_split=2)

# After simulation, collect experiences from all cars
all_experiences = []
for car in model.cars:  # or your list of car agents
    all_experiences.extend(car.experience_buffer)


import numpy as np

gamma = 0.99  # discount factor
actions = [0, 1]  # list of possible actions

# Fitted Q Iteration loop
n_iterations = 10  # Number of FQI iterations
for iteration in range(n_iterations):
    X = []
    y = []
    for (state, action, reward, next_state) in all_experiences:
        next_state_action = [np.concatenate([next_state, [a]]) for a in actions]
        if iteration == 0:
            max_q_next = 0.0  # regressor not fitted yet
        else:
            q_next = q_regressor.predict(next_state_action)
            max_q_next = np.max(q_next)
        q_target = reward + gamma * max_q_next
        X.append(np.concatenate([state, [action]]))
        y.append(q_target)
    X = np.array(X)
    y = np.array(y)
    print(f"Iteration {iteration+1}: X shape = {X.shape}, y shape = {y.shape}")
    q_regressor.fit(X, y)
    print(f"FQI iteration {iteration+1} complete.")

# After Fitted Q Iteration, run evaluation with RL-based actions
print("\nStarting RL-based evaluation...")
eval_steps = 200  # Number of evaluation steps
for step in range(eval_steps):
    model.step(q_regressor=q_regressor)
    collision_count = sum(car.is_colliding for car in model.cars)
    print(f"Step {step}: Collisions = {collision_count}")
print("Evaluation finished.")