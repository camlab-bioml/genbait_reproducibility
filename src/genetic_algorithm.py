import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

# Cache for previously computed fitness values
fitness_cache = {}

# Pre-computed values for original data
original_data_values = {}

def precompute_original_data(df_norm, n_components):
    """
    Precompute and store NMF and other values for the original data.
    """
    original_data = df_norm.to_numpy()
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_original = nmf.fit_transform(original_data)
    basis_matrix_original = nmf.components_.T
    original_data_values['scores_matrix'] = scores_matrix_original
    original_data_values['basis_matrix'] = basis_matrix_original

def evalSubsetCorrelation(df_norm, n_components, subset_range, individual):
    """
    Evaluate the fitness of an individual subset of features.
    """
    individual_key = tuple(individual)  # Convert individual to tuple to use as dictionary key
    
    if individual_key in fitness_cache:
        return fitness_cache[individual_key]  # Retrieve previously computed fitness value

    subset_indices = [i for i in range(len(individual)) if individual[i] == 1]
    
    if not (subset_range[0] <= len(subset_indices) <= subset_range[1]):
        return 0,  # Penalize invalid individuals heavily

    # Subset the data and apply NMF
    subset_data = df_norm.to_numpy()[subset_indices, :]
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Use precomputed values for original data
    scores_matrix_original = original_data_values['scores_matrix']
    basis_matrix_original = original_data_values['basis_matrix']

    # Calculate cosine similarity and cost matrix for Hungarian algorithm
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]
    
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:n_components, n_components:]
    
    # Extract relevant statistics from the correlation matrix
    diagonal = np.diag(corr_matrix)
    diagonal_mean = np.mean(diagonal)
    diagonal_min = np.min(diagonal)
    penalty_factor = 0.5
    num_negative_values = np.sum(diagonal < 0)
    fitness = diagonal_mean - penalty_factor * num_negative_values,

    fitness_cache[individual_key] = fitness  # Cache the computed fitness value
    return fitness


def create_initial_individual(subset_range, n_features):
    """Create an individual with a number of selected features within the subset_range."""
    num_selected_features = random.randint(subset_range[0], subset_range[1])
    selected_indices = random.sample(range(n_features), num_selected_features)
    individual = [0] * n_features
    for idx in selected_indices:
        individual[idx] = 1
    return individual

def run_genetic_algorithm(df_norm, n_components, subset_range, population_size, n_generations, cxpb, mutpb, seed=4):
    random.seed(seed)
    np.random.seed(seed)
    precompute_original_data(df_norm, n_components)

    fitness_cache.clear()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    n_features = df_norm.shape[0]

    # Lambda function to create an individual without arguments
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                     lambda: create_initial_individual(subset_range, n_features))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # toolbox.register("individual", tools.initIterate, creator.Individual, 
    #                  create_initial_individual, subset_range, n_features)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalSubsetCorrelation, df_norm, n_components, subset_range)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof