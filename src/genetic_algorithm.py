# import random
# import numpy as np
# import pandas as pd
# from deap import base, creator, tools, algorithms
# from sklearn.decomposition import NMF
# from scipy.optimize import linear_sum_assignment


# # Cache for previously computed fitness values
# fitness_cache = {}

# def evalSubsetCorrelation(df_norm, n_components, subset_range, individual):
#     """
#     Evaluate the fitness of an individual subset of features.
    
#     Parameters:
#         - df_norm: The normalized data frame
#         - individual: The binary representation of the feature subset
#         - n_components: Number of components to be used in NMF (default: 20)
#         - subset_range: A tuple containing the minimum and maximum allowable sizes for the subset (default: (40, 60))
        
#     Returns:
#         - fitness: A tuple containing the fitness value for the individual
#     """
#     # Convert individual to tuple to use as dictionary key
#     individual_key = tuple(individual)
    
#     # If fitness value has been computed before, retrieve it
#     if individual_key in fitness_cache:
#         return fitness_cache[individual_key]

#     # Determine the subset of features from the individual
#     subset_indices = [i for i in range(len(individual)) if individual[i] == 1]
    
#     # Ensure the subset size is within the desired range
#     if not (subset_range[0] <= len(subset_indices) <= subset_range[1]):
#         fitness = 1e-1000,   # penalize invalid individuals heavily
#     else:
#         # Subset the data based on the indices
#         original_data = df_norm.to_numpy()
#         subset_data = original_data[subset_indices, :]

#         # Apply NMF on both original and subset data
#         nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
#         scores_matrix_original = nmf.fit_transform(original_data)
#         basis_matrix_original = nmf.components_.T
#         scores_matrix_subset = nmf.fit_transform(subset_data)
#         basis_matrix_subset = nmf.components_.T

#         # Calculate cosine similarity and compute cost matrix
#         cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
#         cost_matrix = 1 - cosine_similarity

#         # Use Hungarian algorithm to find the best matching between components
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)
#         basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]
        
#         # Compute the correlation matrix
#         corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:n_components, n_components:]
        
#         # Extract relevant statistics from the correlation matrix
#         diagonal = np.diag(corr_matrix)
#         diagonal_mean = np.mean(diagonal)
#         diagonal_min = np.min(diagonal)
        

#         penalty_factor = 0.5
#         num_negative_values = np.sum(diagonal < 0)
#         penalty = penalty_factor * num_negative_values
#         fitness = diagonal_mean - penalty,


#     # Cache the computed fitness value
#     fitness_cache[individual_key] = fitness

#     return fitness



# def run_genetic_algorithm(df_norm, n_components, subset_range, 
#                           population_size, n_generations, cxpb, mutpb, seed=4):
    
#     random.seed(seed)
#     np.random.seed(seed)

#     # This will store previously computed fitness values
#     fitness_cache = {}

#     # Define the problem
#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # single-objective maximization problem
#     creator.create("Individual", list, fitness=creator.FitnessMax) # type: ignore

#     # Create the toolbox
#     toolbox = base.Toolbox()

#     # Attribute generator: define 'n' possible indices for our subset 
#     n = df_norm.shape[0]
#     toolbox.register("indices", random.randint, 0, 1)

#     # Structure initializers
#     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.indices, n) # type: ignore
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual) # type: ignore

#     # Genetic operators
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
#     toolbox.register("select", tools.selTournament, tournsize=3)
#     toolbox.register("evaluate", evalSubsetCorrelation, df_norm, n_components, subset_range)

#     # Create initial population
#     pop = toolbox.population(n=population_size) # type: ignore

#     # Define the hall-of-fame object
#     hof = tools.HallOfFame(10)

#     # Define statistics to be collected
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)

#     # Execute the genetic algorithm
#     pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, stats=stats, halloffame=hof, verbose=True)

#     return pop, logbook, hof


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

# def run_genetic_algorithm(df_norm, n_components, subset_range, population_size, n_generations, cxpb, mutpb, seed=4):
#     """
#     Run the genetic algorithm.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     precompute_original_data(df_norm, n_components)  # Precompute values for the original data

#     fitness_cache.clear()  # Clear fitness cache

#     # Define the problem
#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMax)

#     # Create the toolbox
#     toolbox = base.Toolbox()
#     n = df_norm.shape[0]
#     # n = int(df_norm.shape[0] / 3)
#     toolbox.register("indices", random.randint, 0, 1)
#     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.indices, n)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#     # Genetic operators
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
#     toolbox.register("select", tools.selTournament, tournsize=3)
#     toolbox.register("evaluate", evalSubsetCorrelation, df_norm, n_components, subset_range)

#     # Create initial population
#     pop = toolbox.population(n=population_size)

#     # Define the hall-of-fame object
#     hof = tools.HallOfFame(10)

#     # Define statistics to be collected
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)

#     # Execute the genetic algorithm
#     pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, stats=stats, halloffame=hof, verbose=True)

#     return pop, logbook, hof

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