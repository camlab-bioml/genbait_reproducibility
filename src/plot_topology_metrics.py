import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import networkx as nx
import scipy.sparse as sp
import json
from matplotlib import cm
import matplotlib
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']



def create_prey_prey_network(df_norm):
    """
    Constructs a prey-prey adjacency matrix based on shared baits using sparse matrices.
    """
    binary_matrix = (df_norm > 0).astype(int)
    sparse_binary_matrix = sp.csr_matrix(binary_matrix.values)
    prey_prey_adj = sparse_binary_matrix.T @ sparse_binary_matrix
    prey_prey_adj = pd.DataFrame(prey_prey_adj.toarray(), index=df_norm.columns, columns=df_norm.columns)
    np.fill_diagonal(prey_prey_adj.values, 0)
    return prey_prey_adj




def analyze_network(prey_prey_adj):
    """
    Calculates topology metrics for the prey-prey network.
    """
    G = nx.from_pandas_adjacency(prey_prey_adj)

    metrics = {}
    metrics['degree_distribution'] = np.mean([val for (_, val) in G.degree()])
    metrics['betweenness_centrality'] = np.mean(list(nx.betweenness_centrality(G, normalized=True).values()))
    # metrics['clustering_coefficient'] = np.mean(list(nx.clustering(G).values()))
    # metrics['eigenvector_centrality'] = np.mean(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
    metrics['graph_density'] = nx.density(G)

    if nx.is_connected(G):
        metrics['network_diameter'] = nx.diameter(G)
        metrics['average_shortest_path_length'] = nx.average_shortest_path_length(G)
    else:
        metrics['network_diameter'] = None
        metrics['average_shortest_path_length'] = None

    return metrics



def save_original_network(df_norm, save_path='network/original_network.json'):
    """
    Calculates and saves the topology metrics for the original dataset.
    """
    if os.path.exists(save_path):
        print(f"Loading original network metrics from {save_path}")
        with open(save_path, 'r') as f:
            return json.load(f)
    else:
        print("Calculating original network metrics...")
        prey_prey_adj = create_prey_prey_network(df_norm)
        original_metrics = analyze_network(prey_prey_adj)
        with open(save_path, 'w') as f:
            json.dump(original_metrics, f)
        print(f"Original network metrics saved to {save_path}")
        return original_metrics


def process_ga_file(args):
    """
    Processes a single GA file to calculate metrics.
    """
    file_name, df_norm, directory = args
    parts = file_name.split('_')
    bait_length = int(parts[3])  # Extract bait length
    seed = int(parts[5].replace('seed', '').replace('.csv', ''))  # Extract seed

    # Read the CSV file and calculate metrics
    file_path = os.path.join(directory, file_name)
    selected_baits_df = pd.read_csv(file_path)
    selected_baits = selected_baits_df.iloc[:, 0].tolist()

    # Create subset of the normalized dataset
    df_subset = df_norm.loc[selected_baits]
    mask = (df_subset != 0).any(axis=0)
    df_subset = df_subset.loc[:, mask]

    # Calculate metrics for the subset network
    prey_prey_adj = create_prey_prey_network(df_subset)
    subset_metrics = analyze_network(prey_prey_adj)

    return bait_length, seed, subset_metrics


def save_ga_subset_network_metrics(df_norm, directory, save_path):
    """
    Calculates and saves the topology metrics for GA subset networks for all bait lengths and all seeds
    using multiprocessing to assign each seed to a separate core.
    """
    if os.path.exists(save_path):
        print(f"Loading GA subset network metrics from {save_path}")
        with open(save_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Calculating GA subset network metrics for all bait lengths and all seeds using multiprocessing...")

        # Prepare arguments for multiprocessing
        file_names = [file_name for file_name in os.listdir(directory) if file_name.endswith('.csv')]
        args = [(file_name, df_norm, directory) for file_name in file_names]

        # Use multiprocessing Pool to process files in parallel
        results_ga = {}
        with Pool(processes=min(10, cpu_count())) as pool:  # Limit to 10 cores
            results = pool.map(process_ga_file, args)

        # Aggregate results
        for bait_length, seed, subset_metrics in results:
            if bait_length not in results_ga:
                results_ga[bait_length] = {}
            for metric, value in subset_metrics.items():
                if metric not in results_ga[bait_length]:
                    results_ga[bait_length][metric] = []
                results_ga[bait_length][metric].append(value)

        # Save the results to a JSON file
        with open(save_path, 'w') as f:
            json.dump(results_ga, f)
        print(f"GA subset network metrics saved to {save_path}")
        return results_ga





def process_random_seed(args):
    """
    Helper function to process a single seed for a given bait length.
    """
    df_norm, df_random_baits_all, bait_length, seed = args
    column = (bait_length - 30) * 10 + seed
    results = {}

    if column < df_random_baits_all.shape[1]:
        # Extract baits for the specific bait length and seed
        selected_baits = df_random_baits_all[column].dropna().tolist()

        # Create subset of the normalized dataset
        df_subset = df_norm.loc[selected_baits]
        mask = (df_subset != 0).any(axis=0)
        df_subset = df_subset.loc[:, mask]

        # Calculate metrics for the subset network
        prey_prey_adj = create_prey_prey_network(df_subset)
        subset_metrics = analyze_network(prey_prey_adj)

        results = {metric: value for metric, value in subset_metrics.items()}
    else:
        print(f"Column {column} for bait length {bait_length} and seed {seed} not found. Skipping...")

    return bait_length, results


def append_results_to_file_incremental(results_file, bait_length, metrics):
    """
    Appends results for a specific bait length to the results file.
    """
    # Check if the file exists
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            json.dump({}, f)  # Create an empty JSON structure

    # Read the existing results
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Initialize metrics for the bait length if not already present
    if str(bait_length) not in data:
        data[str(bait_length)] = {}

    # Append metrics as lists
    for metric, value in metrics.items():
        if metric not in data[str(bait_length)]:
            data[str(bait_length)][metric] = []
        data[str(bait_length)][metric].append(value)

    # Write updated data back to the file
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=4)


def save_random_subset_network_metrics_incremental(df_norm, directory, save_path):
    """
    Calculates and saves the topology metrics for random subset networks for all bait lengths and all seeds incrementally.
    """
    print("Calculating random subset network metrics for all bait lengths and all seeds using multiprocessing...")
    df_random_baits_all = pd.read_csv(os.path.join(directory, 'all_random_baits.csv'), header=None)

    # Create a list of arguments for multiprocessing
    tasks = [(df_norm, df_random_baits_all, bait_length, seed) for bait_length in range(30, 81) for seed in range(10)]

    # Use multiprocessing to process each seed
    with Pool(processes=min(10, cpu_count())) as pool:
        for bait_length, results in pool.imap(process_random_seed, tasks):
            if results:  # Only append if results are not empty
                append_results_to_file_incremental(save_path, bait_length, results)

    print(f"Random subset network metrics incrementally saved to {save_path}")



ml_names = [
            'chi_2', 
            'f_classif', 
            'mutual_info_classif', 
            'lasso', 
            'ridge', 
            'elastic_net', 
            'rf', 
            'gbm', 
            'xgb', 
            'nn'
            ]



def process_ml_seed(args):
    """
    Helper function to process a single seed for a given method and bait length.
    """
    df_norm, seed_file, bait_length, method = args
    results = {}

    # Read the seed-specific CSV file
    methods_df = pd.read_csv(seed_file, index_col=0)

    if method in methods_df.columns:
        # Select the top baits for the current bait length
        selected_baits = methods_df[method].dropna().iloc[:bait_length].tolist()

        if selected_baits:  # Ensure there are selected baits
            df_subset = df_norm.loc[selected_baits]
            mask = (df_subset != 0).any(axis=0)
            df_subset = df_subset.loc[:, mask]

            # Calculate metrics for the subset network
            prey_prey_adj = create_prey_prey_network(df_subset)
            results = analyze_network(prey_prey_adj)
        else:
            print(f"No selected baits for method {method} at bait length {bait_length}. Skipping...")
    else:
        print(f"Method {method} not found in file {seed_file}. Skipping...")

    return bait_length, results


def append_ml_results_to_file(results_file, bait_length, metrics):
    """
    Appends results for a specific bait length to the results file incrementally.
    """
    # Check if the file exists
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            json.dump({}, f)  # Create an empty JSON structure

    # Read the existing results
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Initialize metrics for the bait length if not already present
    if str(bait_length) not in data:
        data[str(bait_length)] = {}

    # Append metrics as lists
    for metric, value in metrics.items():
        if metric not in data[str(bait_length)]:
            data[str(bait_length)][metric] = []
        data[str(bait_length)][metric].append(value)

    # Write updated data back to the file
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=4)



def save_ml_subset_network_metrics_incremental(df_norm, directory, save_path, bait_lengths=range(30, 81), methods=ml_names):
    """
    Calculates and saves the topology metrics for multiple ML methods incrementally for each bait length and seed.

    Parameters:
    - df_norm: Normalized DataFrame containing the full dataset.
    - directory: Path to the directory containing the CSV files for different seeds.
    - save_path: Path to save the resulting JSON file.
    - bait_lengths: Range or list of bait lengths to process.
    - methods: List of ML methods to process.
    """
    print(f"Calculating ML subset network metrics for methods: {methods}")

    seed_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    for method in methods:
        print(f"Processing method: {method}")

        for bait_length in bait_lengths:
            print(f"Processing bait length {bait_length} for method {method}...")

            # Create a list of tasks for multiprocessing
            tasks = [(df_norm, seed_file, bait_length, method) for seed_file in seed_files]

            # Use multiprocessing to process each seed
            with Pool(processes=min(10, cpu_count())) as pool:
                for bait_length, results in pool.imap(process_ml_seed, tasks):
                    if results:  # Only append if results are not empty
                        append_ml_results_to_file(save_path, bait_length, results)

    print(f"ML subset network metrics for methods {methods} saved to {save_path}")






# Function to check for JSON files
def check_and_generate_results(json_path, function_to_run, *args, **kwargs):
    """
    Checks if a JSON file exists. If not, runs the provided function to generate and save the data.
    """
    if os.path.exists(json_path):
        print(f"Loading results from {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        print(f"{json_path} not found. Generating results...")
        results = function_to_run(*args, **kwargs)
        with open(json_path, 'w') as f:
            json.dump(results, f)
        print(f"Results saved to {json_path}")
        return results


def compare_subset_with_original(original_metrics, subset_metrics_path, save_path):
    """
    Compares subset metrics with the original metrics and saves the comparison results as a JSON file.
    """
    # Check if the comparison file already exists
    if os.path.exists(save_path):
        print(f"Loading comparison results from {save_path}")
        with open(save_path, 'r') as f:
            return json.load(f)

    print(f"Comparing subsets from {subset_metrics_path} with original network metrics...")

    # Load subset metrics
    with open(subset_metrics_path, 'r') as f:
        subset_metrics = json.load(f)

    comparison_results = {}

    # Compare each subset with the original metrics
    for bait_length, metrics in subset_metrics.items():
        comparison_results[bait_length] = {}
        for metric, values in metrics.items():  # `values` is a list for multiple seeds
            comparison_results[bait_length][metric] = {
                "absolute_difference": [v - original_metrics[metric] for v in values],
                "relative_change": [
                    (v - original_metrics[metric]) / original_metrics[metric] if original_metrics[metric] != 0 else None
                    for v in values
                ],
                "ratio": [v / original_metrics[metric] if original_metrics[metric] != 0 else None for v in values],
            }

    # Save comparison results to a JSON file
    with open(save_path, 'w') as f:
        json.dump(comparison_results, f)
    print(f"Comparison results saved to {save_path}")
    return comparison_results


def aggregate_and_plot_comparison_results(json_paths, metric, comparison_type, bait_lengths=range(30, 81), save_path='plots/comparison_plot.png'):
    """
    Aggregates data from comparison JSON files, calculates mean values, sorts methods by closeness to 1 (for ratios),
    and creates a boxplot for each method, aggregating data across all bait lengths.
    """
    aggregated_data = []

    # Aggregate data from JSON files
    for method, json_path in json_paths.items():
        with open(json_path, 'r') as f:
            comparison_results = json.load(f)

        if method in ['GA', 'Random']:
            # Handle GA and Random results
            for bait_length in bait_lengths:
                if str(bait_length) in comparison_results:
                    metric_values = comparison_results[str(bait_length)].get(metric, {}).get(comparison_type, [])
                    for value in metric_values:
                        aggregated_data.append({
                            'Method': method,
                            'Value': value
                        })
        elif method == 'ML':
            # Handle ML results
            for ml_method, method_results in comparison_results.items():
                for bait_length in bait_lengths:
                    if str(bait_length) in method_results:
                        metric_values = method_results[str(bait_length)].get(metric, {}).get(comparison_type, [])
                        for value in metric_values:
                            aggregated_data.append({
                                'Method': ml_method,
                                'Value': value
                            })

    # Convert to a DataFrame for plotting
    df = pd.DataFrame(aggregated_data)

    if comparison_type == 'ratio':
        # Calculate the mean for each method and sort by closeness to 1
        method_means = df.groupby('Method')['Value'].mean()
        sorted_methods = method_means.apply(lambda x: abs(x - 1)).sort_values().index.tolist()
    else:
        # Sort methods by descending mean for other comparison types
        method_means = df.groupby('Method')['Value'].mean().sort_values(ascending=False)
        sorted_methods = method_means.index.tolist()

    # Create boxplot with sorted methods
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Method', y='Value', order=sorted_methods, palette='Set2')
    plt.title(f"{metric.capitalize()} ({comparison_type.replace('_', ' ').capitalize()})")
    plt.xlabel("Method")
    plt.ylabel(f"{comparison_type.capitalize()} of {metric.replace('_', ' ').capitalize()}")
    plt.axhline(1, color='gray', linestyle='--', linewidth=1, label='Ideal Ratio = 1')  # Show reference line at ratio = 1
    plt.legend()
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_topology_metrics(df_norm, ga_directory, random_directory, ml_directory, plot_dir, output_dir):

    # df_norm_path = 'datasets/df_norm.csv'
    # ga_directory = 'top_features_GA_seeds'
    # random_directory = 'random_baseline_benchmark'
    # ml_directory = 'ML_results'
    # output_dir = 'network'
    # plot_dir = 'network'

    # Ensure output and plot directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load normalized dataset
    # df_norm = pd.read_csv(df_norm_path, index_col=0)

    # Save original network metrics
    original_metrics_path = os.path.join(output_dir, 'original_network.json')
    if not os.path.exists(original_metrics_path):
        print("Calculating original network metrics...")
        original_metrics = save_original_network(df_norm, save_path=original_metrics_path)
    else:
        print(f"Loading original network metrics from {original_metrics_path}")
        with open(original_metrics_path, 'r') as f:
            original_metrics = json.load(f)

    # Save GA subset network metrics
    ga_metrics_path = os.path.join(output_dir, 'ga_subset_network.json')
    if not os.path.exists(ga_metrics_path):
        print("Calculating GA subset network metrics...")
        save_ga_subset_network_metrics(df_norm, ga_directory, save_path=ga_metrics_path)

    # Save Random subset network metrics
    random_metrics_path = os.path.join(output_dir, 'random_subset_network.json')
    if not os.path.exists(random_metrics_path):
        print("Calculating Random subset network metrics...")
        save_random_subset_network_metrics_incremental(df_norm, random_directory, save_path=random_metrics_path)

    # # Save ML subset network metrics
    ml_metrics_path = os.path.join(output_dir, 'ml_subset_network.json')
    if not os.path.exists(ml_metrics_path):
        print("Calculating ML subset network metrics...")
        save_ml_subset_network_metrics_incremental(df_norm, ml_directory, save_path=ml_metrics_path)

    # Compare GA subset with original network
    ga_comparison_path = os.path.join(output_dir, 'ga_comparison_with_original.json')
    if not os.path.exists(ga_comparison_path):
        print("Comparing GA subset with original metrics...")
        compare_subset_with_original(original_metrics, ga_metrics_path, save_path=ga_comparison_path)

    # Compare Random subset with original network
    random_comparison_path = os.path.join(output_dir, 'random_comparison_with_original.json')
    if not os.path.exists(random_comparison_path):
        print("Comparing Random subset with original metrics...")
        compare_subset_with_original(original_metrics, random_metrics_path, save_path=random_comparison_path)

    # Compare ML subset with original network
    ml_comparison_path = os.path.join(output_dir, 'ml_comparison_with_original.json')
    if not os.path.exists(ml_comparison_path):
        print("Comparing ML subset with original metrics...")
        compare_subset_with_original(original_metrics, ml_metrics_path, save_path=ml_comparison_path)

    # Aggregate and plot comparison results
    comparison_json_paths = {
        'GA': ga_comparison_path,
        'Random': random_comparison_path,
        'ML': ml_comparison_path
    }

    print("Generating plots...")
    
    metrics = ['degree_distribution', 'betweenness_centrality', 'graph_density', 'network_diameter', 'average_shortest_path_length']
    comparison_types = [
        # 'relative_change',
        # 'absolute_difference', 
        'ratio'
        ]

    for metric in metrics:
        for comparison_type in comparison_types:
            aggregate_and_plot_comparison_results(
                json_paths=comparison_json_paths,
                metric=metric,
                comparison_type=comparison_type,
                bait_lengths=range(30, 81),
                save_path=os.path.join(plot_dir, f'{metric}_{comparison_type}_plot.png')
            )

    print("All tasks completed successfully.")


