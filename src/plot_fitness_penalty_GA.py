import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def plot_fitness_diagonal_mean_penalty():
    # Load data from pickle file
    with open('fitness_penalty/best_individuals_per_gen.pkl', 'rb') as f:
        best_individuals_per_gen = pickle.load(f)

    # Prepare data for plotting
    generations = list(best_individuals_per_gen.keys())
    fitness_values = [best_individuals_per_gen[gen]['fitness'] for gen in generations]
    diagonal_means = [best_individuals_per_gen[gen]['diagonal_mean'] for gen in generations]
    penalties = [best_individuals_per_gen[gen]['penalty'] for gen in generations]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_values, label='Fitness')
    plt.plot(generations, diagonal_means, label='Diagonal Mean')
    plt.plot(generations, penalties, label='Penalty')
    
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.title('Evolution of Fitness, Diagonal Mean, and Penalty over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/fitness_penalty_diagonal_mean.png', dpi=300)





def plot_fitness_diagonal_mean_penalty_range():
    subset_ranges = [(36, 56), (37, 57), (38, 58), (39, 59), (40, 60), (41,61), (42,62)]

    for subset_range in subset_ranges:
        upper_bound = subset_range[1]
        file_suffix = f"{upper_bound}"
        
        # Load data from pickle file
        with open(f'fitness_penalty/best_individuals_per_gen{file_suffix}.pkl', 'rb') as f:
            best_individuals_per_gen = pickle.load(f)

        # Prepare data for plotting
        generations = list(best_individuals_per_gen.keys())
        fitness_values = [best_individuals_per_gen[gen]['fitness'] for gen in generations]
        diagonal_means = [best_individuals_per_gen[gen]['diagonal_mean'] for gen in generations]
        penalties = [best_individuals_per_gen[gen]['penalty'] for gen in generations]

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # Adjust the size as needed

        # Plot Fitness
        axs[0].plot(generations, fitness_values, label='Fitness')
        axs[0].set_title(f'Fitness over Generations (Subset Range {subset_range})')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Fitness')
        axs[0].set_ylim(0, 1)
        axs[0].set_xlim(0, 1000)
        axs[0].grid(True)

        # Plot Diagonal Mean
        axs[1].plot(generations, diagonal_means, label='Diagonal Mean', color='orange')
        axs[1].set_title(f'Diagonal Mean over Generations (Subset Range {subset_range})')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Diagonal Mean')
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(0, 1000)
        axs[1].grid(True)

        # Plot Penalty
        axs[2].plot(generations, penalties, label='Penalty', color='green')
        axs[2].set_title(f'Penalty over Generations (Subset Range {subset_range})')
        axs[2].set_xlabel('Generation')
        axs[2].set_ylabel('Penalty')
        axs[2].set_ylim(0, 1)
        axs[2].set_xlim(0, 1000)
        axs[2].grid(True)

        # Adjust the layout
        plt.tight_layout()

        # Save plot with a unique file name
        plt.savefig(f'plots/fitness_penalty_diagonal_mean_{file_suffix}.png', dpi=300)
        plt.close(fig)  # Close the figure to free memory
