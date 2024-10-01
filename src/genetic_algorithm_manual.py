import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import pickle
import concurrent.futures


# Cache for previously computed fitness values
fitness_cache = {}

def evalSubsetCorrelation(df_norm, n_components, subset_range, individual):
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
        fitness = 1e-1000,   # penalize invalid individuals heavily
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
        

        penalty_factor = 0.5
        num_negative_values = np.sum(diagonal < 0)
        penalty = penalty_factor * num_negative_values
        fitness = diagonal_mean - penalty,


    # Cache the computed fitness value
    fitness_cache[individual_key] = fitness

    return fitness


def calculate_diagonal_mean_and_penalty(df_norm, n_components, individual, subset_range):
    # Determine the subset of features from the individual
    subset_indices = [i for i in range(len(individual)) if individual[i] == 1]

    # Ensure the subset size is within the desired range
    if not (subset_range[0] <= len(subset_indices) <= subset_range[1]):
        return None, None  # Can't calculate if the individual is invalid

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
    

    penalty_factor = 0.5
    num_negative_values = np.sum(diagonal < 0)
    penalty = penalty_factor * num_negative_values

    return diagonal_mean, penalty


def run_genetic_algorithm_manual(df_norm, n_components, subset_range, 
                          population_size, n_generations, cxpb, mutpb, seed=4):
    random.seed(seed)
    np.random.seed(seed)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    n = df_norm.shape[0]
    toolbox.register("indices", random.randint, 0, 1)

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.indices, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalSubsetCorrelation, df_norm, n_components, subset_range)

    pop = toolbox.population(n=population_size)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    best_individuals_per_gen = {}

    for gen in range(n_generations):
        print(f"generation: {gen}")
        pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1, stats=stats, halloffame=hof, verbose=True)

        best_ind = tools.selBest(pop, 1)[0]
        best_fitness = best_ind.fitness.values[0]

        diagonal_mean, penalty = calculate_diagonal_mean_and_penalty(df_norm, n_components, best_ind, subset_range)

        best_individuals_per_gen[gen] = {
            "fitness": best_fitness,
            "diagonal_mean": diagonal_mean,
            "penalty": penalty
        }
    # At the end of the function, save the results
    with open('fitness_penalty/hall_of_fame.pkl', 'wb') as f:
        pickle.dump(hof, f)

    with open('fitness_penalty/population.pkl', 'wb') as f:
        pickle.dump(pop, f)

    with open('fitness_penalty/best_individuals_per_gen.pkl', 'wb') as f:
        pickle.dump(best_individuals_per_gen, f)

    return None




def run_for_one_subset_range(df_norm, n_components, population_size, n_generations, cxpb, mutpb, seed, subset_range):
    upper_bound = subset_range[1]
    file_suffix = f"{upper_bound}"
    
    random.seed(seed)
    np.random.seed(seed)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    n = df_norm.shape[0]
    toolbox.register("indices", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.indices, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalSubsetCorrelation, df_norm, n_components, subset_range)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    best_individuals_per_gen = {}

    for gen in range(n_generations):
        print(f"Generation: {gen}, Subset Range: {subset_range}")
        pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1, stats=stats, halloffame=hof, verbose=True)

        best_ind = tools.selBest(pop, 1)[0]
        best_fitness = best_ind.fitness.values[0]
        diagonal_mean, penalty = calculate_diagonal_mean_and_penalty(df_norm, n_components, best_ind, subset_range)
        best_individuals_per_gen[gen] = {"fitness": best_fitness, "diagonal_mean": diagonal_mean, "penalty": penalty}

    with open(f'fitness_penalty/hall_of_fame{file_suffix}.pkl', 'wb') as f:
        pickle.dump(hof, f)
    with open(f'fitness_penalty/population{file_suffix}.pkl', 'wb') as f:
        pickle.dump(pop, f)
    with open(f'fitness_penalty/best_individuals_per_gen{file_suffix}.pkl', 'wb') as f:
        pickle.dump(best_individuals_per_gen, f)

def run_genetic_algorithm_manual_range(df_norm, n_components, population_size, n_generations, cxpb, mutpb, seed=4):
    subset_ranges = [(36, 56), (37, 57), (38, 58), (39, 59), (40, 60), (41, 61), (42,62)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_for_one_subset_range, df_norm, n_components, population_size, n_generations, cxpb, mutpb, seed, subset_range) 
                   for subset_range in subset_ranges]

        concurrent.futures.wait(futures)

    return None