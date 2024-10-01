import os
import pickle
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from genetic_algorithm import evalSubsetCorrelation, run_genetic_algorithm, fitness_cache



# Function to save an individual component (population, logbook, or hof)
def save_component(component, filename):
    with open(filename, 'wb') as f:
        pickle.dump(component, f)

def run_ga_for_numbers(df_norm, n_components, population_size, n_generations, cxpb, mutpb):
    results_dir = "GA_number_of_baits"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for num_features in range(10, 61):
        # Clear the fitness cache at the start of each run
        fitness_cache.clear()

        subset_range = (num_features, num_features+20)
        pop, logbook, hof = run_genetic_algorithm(df_norm, n_components, subset_range,
                                                  population_size, n_generations, cxpb, mutpb)
        
        # Save the results
        save_component(pop, os.path.join(results_dir, f"popfile{num_features+20}.pkl"))
        save_component(logbook, os.path.join(results_dir, f"logbookfile{num_features+20}.pkl"))
        save_component(hof, os.path.join(results_dir, f"hoffile{num_features+20}.pkl"))

        print(f"Results for {num_features+20} features saved.")
