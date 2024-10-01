import os
import pickle
import random
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from deap import base, creator, tools, algorithms
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from genetic_algorithm import evalSubsetCorrelation, run_genetic_algorithm, fitness_cache
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

# Function to save an individual component (population, logbook, or hof)
def save_component(component, filename):
    with open(filename, 'wb') as f:
        pickle.dump(component, f)

def run_ga_with_seed(df_norm, n_components, population_size, n_generations, cxpb, mutpb, num_features, seed, file_path):
    results_dir = file_path
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)



    # Clear the fitness cache at the start of each run
    fitness_cache.clear()

    subset_range = (num_features-10, num_features)
    pop, logbook, hof = run_genetic_algorithm(df_norm, n_components, subset_range,
                                              population_size, n_generations, cxpb, mutpb, seed)

    # Save the results with seed and feature number in the filename
    save_component(pop, os.path.join(results_dir, f"popfile_features_{num_features}_seed_{seed}.pkl"))
    save_component(logbook, os.path.join(results_dir, f"logbookfile_features_{num_features}_seed_{seed}.pkl"))
    save_component(hof, os.path.join(results_dir, f"hoffile_features_{num_features}_seed_{seed}.pkl"))

    print(f"Results for {num_features} features with seed {seed} saved.")
    return f"Completed seed {seed} for feature count {num_features}"

def run_ga_for_number_of_baits_and_seeds(df_norm, n_components, population_size, n_generations, cxpb, mutpb, file_path):
    for num_features in range(30, 81):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_ga_with_seed, df_norm, n_components, population_size, n_generations, cxpb, mutpb, num_features, seed, file_path) for seed in range(10)]
            for future in concurrent.futures.as_completed(futures):
                print(future.result())


