import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import warnings
import os
warnings.filterwarnings('ignore')



fitness_cache = {}

def evalSubsetCorrelationRandom(df_norm, n_components, subset_range, individual, use_gradient_penalty=False):
    """
    Evaluate the fitness of an individual subset of features.
    
    Parameters:
        - df_norm: The normalized data frame
        - individual: The binary representation of the feature subset
        - n_components: Number of components to be used in NMF (default: 20)
        - subset_range: A tuple containing the minimum and maximum allowable sizes for the subset (default: (40, 60))
        
    Returns:
        - fitness: A tuple containing the fitness value for the individual
    """
    # Convert individual to tuple to use as dictionary key
    individual_key = tuple(individual)
    
    # If fitness value has been computed before, retrieve it
    if individual_key in fitness_cache:
        return fitness_cache[individual_key]

    # Determine the subset of features from the individual
    subset_indices = [i for i in range(len(individual)) if individual[i] == 1]
    
    # Ensure the subset size is within the desired range
    if not (subset_range[0] <= len(subset_indices) <= subset_range[1]):
        fitness = 0,   # penalize invalid individuals heavily
    else:
        # Subset the data based on the indices
        original_data = df_norm.to_numpy()
        subset_data = original_data[subset_indices, :]

        # Apply NMF on both original and subset data
        nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
        scores_matrix_original = nmf.fit_transform(original_data)
        basis_matrix_original = nmf.components_.T
        scores_matrix_subset = nmf.fit_transform(subset_data)
        basis_matrix_subset = nmf.components_.T

        # Calculate cosine similarity and compute cost matrix
        cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
        cost_matrix = 1 - cosine_similarity

        # Use Hungarian algorithm to find the best matching between components
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]
        
        # Compute the correlation matrix
        corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:n_components, n_components:]
        
        # Extract relevant statistics from the correlation matrix
        diagonal = np.diag(corr_matrix)
        diagonal_mean = np.mean(diagonal)
        diagonal_min = np.min(diagonal)
        
        
        if use_gradient_penalty:
            penalty_factor = 0.5
            num_negative_values = np.sum(diagonal < 0)
            fitness = diagonal_mean - penalty_factor * num_negative_values,
        else:
            fitness = diagonal_mean,

    # Cache the computed fitness value
    fitness_cache[individual_key] = fitness

    return fitness




def generate_random_baseline(df_norm, n_components, subset_range, file_path):
    def generate_random_subset(indices, subset_range):
        subset_length = np.random.randint(subset_range[0], subset_range[1] + 1)
        return np.random.choice(indices, size=subset_length, replace=False)

    # Generate 1000 random subsets
    random_subsets = [generate_random_subset(df_norm.index, subset_range) for _ in range(1000)]

    # Convert the list of numpy arrays to a list of lists
    random_subsets_lists = [subset.tolist() for subset in random_subsets]

    # Prepare data for saving the indices
    random_subsets_indices = [[df_norm.index.get_loc(index) for index in subset] for subset in random_subsets]

    fitness_values = []
    best_baits = None
    max_fitness = float('-inf')  # Initialize the max fitness to a very low value

    # Evaluate each subset using the evalSubsetCorrelation function
    for subset in random_subsets_lists:
        # Create an individual as a binary representation of the feature subset
        individual = [0] * len(df_norm)
        for idx in subset:
            individual[df_norm.index.get_loc(idx)] = 1  # Mark feature as selected
        
        # Evaluate the fitness of the individual
        fitness = evalSubsetCorrelationRandom(df_norm, n_components, subset_range, individual)
        fitness_values.append(fitness)

        # Track the best baits corresponding to the highest fitness score
        if fitness[0] > max_fitness:
            max_fitness = fitness[0]
            best_baits = subset  # Update best baits when a new max fitness is found

    # Save the fitness values, subsets and indices to CSV if needed
    fitness_df = pd.DataFrame(fitness_values)
    fitness_df.to_csv(f'{file_path}random_subsets_fitness.csv', index=False, header=False)

    subsets_df = pd.DataFrame(random_subsets_lists).T  # Transpose so each subset is a column
    subsets_df.to_csv(f'{file_path}all_random_baits.csv', index=False, header=False)

    indices_df = pd.DataFrame(random_subsets_indices).T
    indices_df.to_csv(f'{file_path}random_subsets_indices.csv', index=False, header=False)

    return fitness_values, best_baits  # Return both the fitness values and the best baits


def generate_best_set_sequence_from_random(fitnesses):
    best_set = []
    max_fitness = float('-inf')
    for fitness in fitnesses:
        if fitness > max_fitness:
            max_fitness = fitness
        best_set.append(max_fitness)
    return best_set

def save_best_baits(best_baits, file_path):
    pd.DataFrame({'Best Baits': best_baits}).to_csv(f'{file_path}best_baits.csv', index=False)


def generate_random_baseline_benchmark(df_norm, n_components, file_path):
    # Helper function to generate a random subset of indices with a specific length
    def generate_random_subset(indices, subset_length):
        return np.random.choice(indices, size=subset_length, replace=False)

    # Initialize containers for subsets, indices, and fitness values
    all_subsets_lists = []
    all_subsets_indices = []
    bait_lengths = range(30, 81)  # Bait numbers from 30 to 80
    num_seeds = 10  # Number of seeds (random generations) for each bait number
    fitness_values = {length: [] for length in bait_lengths}  # Dict to store fitness values for each bait number

    # Generate 10 subsets for each length from 30 to 80
    for length in bait_lengths:
        for _ in range(num_seeds):
            subset = generate_random_subset(df_norm.index, length)
            all_subsets_lists.append(subset.tolist())  # Store the subset list

            # Convert indices for the subset
            subset_indices = [df_norm.index.get_loc(index) for index in subset]
            all_subsets_indices.append(subset_indices)

            # Evaluate the fitness of the subset
            individual = [0] * len(df_norm)
            for idx in subset:
                individual[df_norm.index.get_loc(idx)] = 1
            fitness = evalSubsetCorrelationRandom(df_norm, n_components, (30, 80), individual)  # Updated function call
            fitness_values[length].append(fitness)



    # Convert fitness values dict to DataFrame for saving
    fitness_df = pd.DataFrame.from_dict(fitness_values, orient='index').transpose()

    # Extract numeric values from tuples
    for col in fitness_df.columns:
        fitness_df[col] = fitness_df[col].apply(lambda x: x[0] if isinstance(x, tuple) and len(x) == 1 else x)

    # Save the DataFrame as a pickle file
    fitness_df.to_pickle(f'{file_path}random_subsets_fitness.pkl')
    # fitness_df.to_pickle('plots/boxplot_random.pkl')

    # Save the subsets (values) to a single CSV, each subset in its own column
    subsets_df = pd.DataFrame(all_subsets_lists).transpose()
    subsets_df.to_csv(f'{file_path}all_random_baits.csv', index=False, header=False)

    # Save the indices to a single CSV, each subset in its own column
    indices_df = pd.DataFrame(all_subsets_indices).transpose()
    indices_df.to_csv(f'{file_path}random_subsets_indices.csv', index=False, header=False)
