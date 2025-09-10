import argparse
from model import Renderer
from qlearner import QLearner
import pickle

def train():
    actions = ["frenar", "mantener", "acelerar"]
    q_learner = QLearner(actions=actions)
    model = Renderer({'steps': 1000}) # You can adjust the number of steps per episode

    num_episodes = 100 # You can change the number of episodes

    for episode in range(num_episodes):
        model.setup() # Reset the model for a new episode
        
        # We need to get the car from the model to interact with it
        # Assuming we are training the first car for simplicity
        if not model.cars:
            print("No cars in the model to train.")
            return
            
        car_to_train = model.cars[0]
        car_to_train.q_learner = q_learner

        for step in range(model.p.steps):
            # Get current state
            state = model.get_state(car_to_train)

            # Choose action
            action = q_learner.choose_action(state)

            # Apply action
            car_to_train.apply_q_action(action)

            # Step the model
            model.step()

            # Get new state
            new_state = model.get_state(car_to_train)

            # Compute reward
            reward = car_to_train.compute_reward(car_to_train)

            # Update Q-table
            q_learner.update(state, action, reward, new_state)
        
        print(f"Episode {episode + 1}/{num_episodes} completed.")

    # Save the Q-table
    q_learner.save('q_table.pkl')
    print("Training finished and Q-table saved to q_table.pkl")

if __name__ == "__main__":
    train()
