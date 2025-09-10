import argparse
from model import Renderer
from training_model import train
from qlearner import QLearner
import pickle

def main():
    parser = argparse.ArgumentParser(description="Run simulation or training.")
    parser.add_argument('--train', action='store_true', help='Run the training script.')
    parser.add_argument('--q-table', type=str, help='Path to the Q-table file to use for the simulation.')
    args = parser.parse_args()

    if args.train:
        train()
    else:
        model = Renderer({'steps': 3000})
        if args.q_table:
            q_learner = QLearner(actions=["frenar", "mantener", "acelerar"])
            try:
                q_learner.load(args.q_table)
                print(f"Q-table loaded from {args.q_table}")
                # If a q_table is loaded, we assume we want to use it to control the car
                # We need to modify the model's step logic to use the q_learner
                # For now, we just load it. The logic to use it in the simulation
                # would need to be added to the model.step() or a similar method.
                # This is a placeholder for that logic.
                if model.cars:
                    car_to_control = model.cars[0]
                    # In a real scenario, you would get the state, choose an action with epsilon=0,
                    # and apply it in each step of the simulation.
                    # This requires modifying the Renderer class, which is out of scope for now.
                    print("Running simulation with loaded Q-table (Note: car control logic not fully implemented in simulation).")

            except FileNotFoundError:
                print(f"Error: Q-table file not found at {args.q_table}")
                return
        
        model.run()

if __name__ == "__main__":
    main()
