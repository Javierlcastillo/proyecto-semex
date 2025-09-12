import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison(fixed_csv, qlearning_csv):
    """
    Reads two CSV files and plots their data on a line graph for comparison.

    Args:
        fixed_csv (str): Path to the CSV file for the 'fixed' method.
        qlearning_csv (str): Path to the CSV file for the 'q-learning' method.
    """
    try:
        # Read the data from the CSV files. Assuming no header.
        df_fixed = pd.read_csv(fixed_csv, header=None, names=['finished_cars'])
        df_qlearning = pd.read_csv(qlearning_csv, header=None, names=['finished_cars'])

        # Create the plot
        plt.figure(figsize=(12, 7))

        # Plot data for both files
        plt.plot(df_fixed.index, df_fixed['finished_cars'], label='Fixed', color='blue')
        plt.plot(df_qlearning.index, df_qlearning['finished_cars'], label='Q-Learning', color='orange')

        # Add titles and labels
        plt.title('Comparison of Finished Cars per Step: Fixed vs. Q-Learning')
        plt.xlabel('Steps')
        plt.ylabel('Number of Finished Cars')
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the CSV files exist in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     # Define the names of your CSV files
#     fixed_file = 'cars_per_stepfixed.csv'
#     qlearning_file = 'cars_per_stepqlearning.csv'
    
#     plot_comparison(fixed_file, qlearning_file)
