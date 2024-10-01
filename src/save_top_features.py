import pandas as pd
import os
from data_storage import load_genetic_algorithm_results

def get_and_save_top_features_from_ga(hof, df_norm, population_size, generation_count, save_path):
    """
    Save top lists from the GA's hall of fame to CSV files and return the lists.

    Args:
    - hof (list): Hall of fame (top N individuals).
    - df_norm (DataFrame): Normalized data used in GA.
    - feature_names (list): List of feature names.
    - population_size (int): Population size used in the GA.
    - generation_count (int): Number of generations used in the GA.

    Returns:
    - List of lists containing top features from GA.
    """
    feature_names = df_norm.index.tolist()

    top_features_lists = []

    for i, individual in enumerate(hof):
        subset_indices = [index for index, value in enumerate(individual) if value == 1]
        subset_features = [feature_names[i] for i in subset_indices]
        
        top_features_lists.append(subset_features)

        df = pd.DataFrame(subset_features, columns=['selected_baits'])
        df.to_csv(f'{save_path}/top {i+1} selected features GA pop{population_size} gen{generation_count}.csv', index=False)

    return top_features_lists



def get_and_save_top_features_from_ga_seeds(df_norm, ga_dir, save_dir, top_n=1):
    """
    Process all GA results and save the top features to CSV files.

    Args:
    - ga_dir (str): Directory containing GA result files.
    - save_dir (str): Directory to save the top features CSV files.
    - df_norm (DataFrame): Normalized data used in GA.
    - top_n (int): Number of top features to save.
    """
    for filename in os.listdir(ga_dir):
        if 'hoffile' in filename and filename.endswith('.pkl'):
            # Extract features and seed number from filename
            parts = filename.split('_')
            features = int(parts[2])  # Assuming 'features' is always the third element
            seed = int(parts[4].split('.')[0])  # Assuming 'seed' is always the fifth element

            # Load GA results
            pop, logbook, hof = load_genetic_algorithm_results(
                os.path.join(ga_dir, f'popfile_features_{features}_seed_{seed}.pkl'),
                os.path.join(ga_dir, f'logbookfile_features_{features}_seed_{seed}.pkl'),
                os.path.join(ga_dir, f'hoffile_features_{features}_seed_{seed}.pkl'))

            # Process and save top features
            os.makedirs(save_dir, exist_ok=True)

            feature_names = df_norm.index.tolist()  
            for i, individual in enumerate(hof):
                if i >= top_n:  # Save only top_n features
                    break

                subset_indices = [index for index, value in enumerate(individual) if value == 1]
                subset_features = [feature_names[index] for index in subset_indices]

                if len(subset_features) == features:
                    df = pd.DataFrame(subset_features, columns=['selected_features'])
                    file_name = f'top_{i+1}_features_{features}_seed_{seed}.csv'
                    df.to_csv(os.path.join(save_dir, file_name), index=False)
