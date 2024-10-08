import pandas as pd
import numpy as np
import os
import igraph as ig
import leidenalg
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import pickle
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from matplotlib import cm
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

ml_names = ['chi_2', 'f_classif', 'mutual_info_classif', 'lasso', 'ridge', 'elastic_net', 'rf', 'gbm', 'xgb']


def get_gmm_hard_assignments(data, n_clusters, seed):
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    gmm.fit(data)
    return gmm.predict(data)

def precalculate_gmm_assignments_for_original(df_original, cluster_numbers, seed):
    original_assignments = {}
    data = df_original.transpose().values  # Transposing so samples are rows
    for n_clusters in cluster_numbers:
        original_assignments[n_clusters] = get_gmm_hard_assignments(data, n_clusters, seed)
    return original_assignments

def calculate_ari_between_assignments(original_assignments, df_original, df_subset, n_clusters, seed):
    common_preys = df_original.columns.intersection(df_subset.columns)
    df_subset_common = df_subset[common_preys].transpose().values
    subset_assignments = get_gmm_hard_assignments(df_subset_common, n_clusters, seed)
    # Only compare the common samples' assignments
    common_original_assignments = original_assignments[n_clusters][np.isin(df_original.columns, common_preys)]
    common_subset_assignments = subset_assignments
    return adjusted_rand_score(common_original_assignments, common_subset_assignments)

def process_ga_results(df_norm, original_assignments, cluster_numbers, ga_path, seed=4):
    random.seed(seed)
    results_ga = {}
    directory = ga_path
    feature_groups = {}

    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            parts = file_name.split('_')
            num_features = int(parts[3])
            if num_features not in feature_groups:
                feature_groups[num_features] = []
            feature_groups[num_features].append(file_name)

    for num_features, files in feature_groups.items():
        results_ga[num_features] = {cluster_number: [] for cluster_number in cluster_numbers}
        for file_name in files:
            selected_baits_df = pd.read_csv(os.path.join(directory, file_name))
            selected_baits = selected_baits_df.iloc[:, 0].tolist()
            df_subset = df_norm.loc[selected_baits]
            mask = (df_subset != 0).any(axis=0)
            df_subset = df_subset.loc[:, mask]
            for cluster_number in cluster_numbers:
                ari = calculate_ari_between_assignments(original_assignments, df_norm, df_subset, cluster_number, seed)
                results_ga[num_features][cluster_number].append(ari)
    
    return results_ga

def process_random_baseline(df_norm, original_assignments, cluster_numbers, random_path, seed=4):
    random.seed(seed)
    results_random = {length: {cluster_number: [] for cluster_number in cluster_numbers} for length in range(30, 81)}
    
    df_random_baits_all = pd.read_csv(f'{random_path}all_random_baits.csv', header=None)
    for bait_length in range(30, 81):  # Adjust as needed
        for column in range((bait_length - 30) * 10, (bait_length - 30) * 10 + 10):  # Adjust the calculation as necessary
            selected_baits = df_random_baits_all[column].dropna().tolist()
            df_subset = df_norm.loc[selected_baits]
            mask = (df_subset != 0).any(axis=0)
            df_subset = df_subset.loc[:, mask]
            for cluster_number in cluster_numbers:
                ari = calculate_ari_between_assignments(original_assignments, df_norm, df_subset, cluster_number, seed)
                results_random[bait_length][cluster_number].append(ari)
    
    return results_random




def process_ml_results(df_norm, original_assignments, cluster_numbers, ml_path, seed=4):
    random.seed(seed)
    results_ml = {method: {entity_count: {cluster_number: [] for cluster_number in cluster_numbers} for entity_count in range(30, 81)} for method in ml_names}
    
    for file_name in os.listdir(ml_path):
        if file_name.endswith('.csv'):
            methods_df = pd.read_csv(os.path.join(ml_path, file_name))
            for method in ml_names:
                for entity_count in range(30, 81):  # Adjust as needed
                    if method in methods_df:
                        selected_baits = methods_df[method].dropna().iloc[:entity_count].tolist()
                        df_subset = df_norm.loc[selected_baits]
                        mask = (df_subset != 0).any(axis=0)
                        df_subset = df_subset.loc[:, mask]
                        for cluster_number in cluster_numbers:
                            ari = calculate_ari_between_assignments(original_assignments, df_norm, df_subset, cluster_number, seed)
                            results_ml[method][entity_count][cluster_number].append(ari)
    
    return results_ml



def analyze_for_seed(df_norm, cluster_numbers, seed, ga_path, ml_path, random_path):
    np.random.seed(seed)
    random.seed(seed)
    original_assignments = precalculate_gmm_assignments_for_original(df_norm, cluster_numbers, seed=4)
    ga_results = process_ga_results(df_norm, original_assignments, cluster_numbers, ga_path, seed)
    random_results = process_random_baseline(df_norm, original_assignments, cluster_numbers, random_path, seed)
    ml_results = process_ml_results(df_norm, original_assignments, cluster_numbers, ml_path, seed)
    
    return {
        'ga': ga_results,
        'random': random_results,
        'ml': ml_results,
    }


def run_parallel_analysis(df_norm, cluster_numbers, seeds, results_file, ga_path, ml_path, random_path):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_for_seed, df_norm, cluster_numbers, seed, ga_path, ml_path, random_path) for seed in seeds]
        
        # Initialize the structure to hold aggregated results.
        aggregated_results = {
            'GA': {},
            'Random': {},
            'ML': {}
        }
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            
            # Aggregating GA results
            for num_features, res_dict in result['ga'].items():
                if num_features not in aggregated_results['GA']:
                    aggregated_results['GA'][num_features] = {}
                for resolution, scores in res_dict.items():
                    aggregated_results['GA'][num_features].setdefault(resolution, []).extend(scores)
            
            # Aggregating Random results
            for length, res_dict in result['random'].items():
                if length not in aggregated_results['Random']:
                    aggregated_results['Random'][length] = {}
                for resolution, scores in res_dict.items():
                    aggregated_results['Random'][length].setdefault(resolution, []).extend(scores)
            
            # Aggregating ML results
            for method, entity_dict in result['ml'].items():
                if method not in aggregated_results['ML']:
                    aggregated_results['ML'][method] = {}
                for entity_count, res_dict in entity_dict.items():
                    if entity_count not in aggregated_results['ML'][method]:
                        aggregated_results['ML'][method][entity_count] = {}
                    for resolution, scores in res_dict.items():
                        aggregated_results['ML'][method][entity_count].setdefault(resolution, []).extend(scores)
    
    # Saving the aggregated results to a file
    with open(results_file, 'wb') as f:
        pickle.dump(aggregated_results, f)

    print(f"Analysis completed. Results saved to '{results_file}'.")



def plot_gmm_hard_ari(df_norm, cluster_numbers, ga_path, ml_path, random_path, save_path, seeds=range(10)):
    results_file=f'{save_path}gmm_hard_results.pkl'
    # Check if the pickle file exists
    if not os.path.exists(results_file):
        run_parallel_analysis(df_norm, cluster_numbers, seeds, results_file, ga_path, ml_path, random_path)
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'red']
    ml_methods = list(results['ML'].keys())
    method_names = ['GA', 'Random'] + ml_methods

    plt.figure(figsize=(12, 6))
    total_width = 0.8  # The total width of the group of boxplots at each resolution
    single_width = total_width / len(method_names)
    offset = np.linspace(-total_width/2, total_width/2, len(method_names))

    tick_positions = []
    spacing_factor = 2  # Increase this factor to increase spacing between groups

    for res_index, resolution in enumerate(cluster_numbers):
        group_center = res_index * spacing_factor * len(method_names) * single_width

        for method_index, method in enumerate(method_names):
            if method in ['GA', 'Random']:
                ari_scores = [score for num_features in results[method].values() for score in num_features.get(resolution, [])]
            else:  # ML method
                ari_scores = [score for num_features in results['ML'][method].values() for score in num_features.get(resolution, [])]

            position = group_center + offset[method_index]

            plt.boxplot(ari_scores, positions=[position], widths=single_width*0.9, patch_artist=True,
                        boxprops=dict(facecolor=colors[method_index % len(colors)]))

        tick_positions.append(group_center)

    plt.xticks(tick_positions, [str(r) for r in cluster_numbers])
    plt.xlabel('Resolution')
    plt.ylabel('ARI Score')
    plt.title('ARI Scores by Method and Resolution')
    plt.legend([plt.Line2D([0], [0], color=colors[i], lw=4) for i, method in enumerate(method_names)], method_names, bbox_to_anchor=(1.05, 1), loc='upper left', )
    plt.tight_layout()
    plt.savefig(f'{save_path}GMM ARI values vs. resolutions.png', dpi=300)
    plt.savefig(f'{save_path}GMM ARI values vs. resolutions.svg', dpi=300)
    plt.clf()


    # Generate shades of blue from the 'Blues' colormap
    n_resolutions = len(cluster_numbers)
    color_map = cm.get_cmap('Blues', n_resolutions + 1)  # '+1' to avoid the lightest color, which is hard to see
    resolution_colors = [color_map(i) for i in range(1, n_resolutions + 1)]
    method_names = ['GA', 'Random'] + ['ML_' + method for method in ml_names]  # Combined method names
    color_dict = dict(zip(cluster_numbers, resolution_colors))

    # Prepare data for plotting
    plot_data = {method: {res: [] for res in cluster_numbers} for method in method_names}

    # Extract data for each method and resolution
    for method_group, methods in results.items():
        if method_group == 'ML':
            for method, method_data in methods.items():
                for num_features, res_data in method_data.items():
                    for res, ari_scores in res_data.items():
                        plot_data['ML_' + method][res].extend(ari_scores)
        else:  # GA or Random
            for num_features, res_data in methods.items():
                for res, ari_scores in res_data.items():
                    plot_data[method_group][res].extend(ari_scores)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    positions = np.arange(len(method_names))

    for i, res in enumerate(cluster_numbers):
        offsets = np.linspace(-0.2, 0.2, len(cluster_numbers))
        for method_index, method in enumerate(method_names):
            pos = positions[method_index] + offsets[i]
            ax.boxplot(plot_data[method][res], positions=[pos], widths=0.12, patch_artist=True,
                    boxprops=dict(facecolor=color_dict[res]))

    # Customize the axes
    ax.set_xticks(positions)
    ax.set_xticklabels([method.replace('ML_', '') for method in method_names], rotation=90)
    ax.set_ylabel('ARI Scores')
    ax.set_title('GMM ARI values vs. each method')

    # Create a legend for the resolutions
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_dict[res], label=f'Resolution {res}') for res in cluster_numbers]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1,1), title="Resolutions")

    plt.tight_layout()
    plt.savefig(f'{save_path}GMM ARI values vs. each method.png', dpi=300)
    plt.savefig(f'{save_path}GMM ARI values vs. each method.svg', dpi=300)
    plt.clf()


    # Initialize figure and subplots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(30, 20), constrained_layout=True)
    axes = axes.flatten()  # Flatten for easy iteration

    # Gather global min and max ARI values for consistent y-axis limits
    all_values = []
    for method_group, methods in results.items():
        for method, method_data in (methods.items() if method_group == 'ML' else [(method_group, methods)]):
            for num_features, res_data in method_data.items():
                for res, ari_scores in res_data.items():
                    all_values.extend(ari_scores)
    global_min, global_max = min(all_values), max(all_values)

    # Plot data for each method
    for idx, method in enumerate(method_names):
        ax = axes[idx]
        if method.startswith('ML_'):
            method_data = results['ML'][method[3:]]  # Trim 'ML_' prefix
        else:
            method_data = results[method]

        boxplot_data = []
        feature_numbers = sorted(method_data.keys())
        for num_features in feature_numbers:
            # Aggregate ARI scores across all resolutions
            ari_scores = [score for res in cluster_numbers for score in method_data[num_features].get(res, [])]
            boxplot_data.append(ari_scores)
        
        ax.boxplot(boxplot_data, positions=range(len(feature_numbers)))
        ax.set_title(method if not method.startswith('ML_') else method[3:])  # Remove 'ML_' prefix for title
        ax.set_xticks(range(len(feature_numbers)))
        ax.set_xticklabels(feature_numbers)
        ax.set_ylim(global_min, global_max)  # Consistent Y-axis range for all subplots
        ax.set_xlabel('Number of Baits')
        ax.set_ylabel('ARI Scores')

    # Hide any unused axes
    for i in range(len(method_names), len(axes)):
        axes[i].axis('off')

    # Adjust layout and save the figure
    plt.suptitle('ARI Scores by Number of Features for Each Method')
    plt.savefig(f'{save_path}GMM ARI values vs. number of baits.png', dpi=300)
    plt.savefig(f'{save_path}GMM ARI values vs. number of baits.svg', dpi=300)
    plt.clf()


        # Initialize figure for plotting
    fig, ax = plt.subplots(figsize=(15, 9))

    # Define a color map for the methods
    colors = plt.cm.tab20(np.linspace(0, 1, len(method_names)))

    # Function to aggregate ARI scores across resolutions for a given method's data
    def aggregate_ari_scores(method_data, resolutions):
        aggregated_scores = {}
        for num_features, res_data in method_data.items():
            all_scores = []
            for res in resolutions:
                all_scores.extend(res_data.get(res, []))
            aggregated_scores[num_features] = all_scores
        return aggregated_scores

    # Plot data for each method
    for idx, method in enumerate(method_names):
        # Determine if method is ML and extract its data accordingly
        if method.startswith('ML_'):
            method_data = results['ML'][method[3:]]  # Trim 'ML_' prefix
        else:
            method_data = results[method]

        # Aggregate ARI scores across all resolutions
        aggregated_scores = aggregate_ari_scores(method_data, cluster_numbers)
        
        # Sort feature numbers for plotting
        feature_numbers = sorted(aggregated_scores.keys())
        means = [np.mean(aggregated_scores[num_features]) for num_features in feature_numbers]
        std_devs = [np.std(aggregated_scores[num_features]) for num_features in feature_numbers]
        
        # Plot mean ARI scores with shaded areas for standard deviation
        ax.plot(feature_numbers, means, label=method if not method.startswith('ML_') else method[3:], color=colors[idx])
        ax.fill_between(feature_numbers, [m - s for m, s in zip(means, std_devs)], [m + s for m, s in zip(means, std_devs)], color=colors[idx], alpha=0.2)

    # Set plot parameters
    ax.set_xlabel('Number of Baits')
    ax.set_ylabel('Aggregated ARI Scores')
    ax.set_title('GMM ARI Scores by Number of Baits for Each Method')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    ax.set_xlim(left=min(feature_numbers), right=max(feature_numbers))
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{save_path}GMM ARI values vs. number of baits_mean shaded.png', dpi=300)
    plt.savefig(f'{save_path}GMM ARI values vs. number of baits_mean shaded.svg', dpi=300)
    plt.clf()


    # Aggregate ARI scores for each method
    method_aggregated_scores = {}
    method_medians = {}
    for method_group, methods in results.items():
        if method_group == 'ML':
            for method, method_data in methods.items():
                all_scores = []
                for num_features, res_data in method_data.items():
                    for res, ari_scores in res_data.items():
                        all_scores.extend(ari_scores)
                method_aggregated_scores['ML_' + method] = all_scores
                method_medians['ML_' + method] = np.median(all_scores)
        else:  # GA or Random
            all_scores = []
            for num_features, res_data in methods.items():
                for res, ari_scores in res_data.items():
                    all_scores.extend(ari_scores)
            method_aggregated_scores[method_group] = all_scores
            method_medians[method_group] = np.median(all_scores)

    # Sort methods by median ARI score in decreasing order
    sorted_methods = sorted(method_medians, key=method_medians.get, reverse=True)

    # Plot
    plt.figure(figsize=(12, 9))
    positions = range(len(sorted_methods))
    boxplot_data = [method_aggregated_scores[method] for method in sorted_methods]
    boxprops = dict(color="black", facecolor="lightblue")  # Define box properties for blue color
    whiskerprops = dict(color="black")
    capprops = dict(color="black")
    medianprops = dict(color="orange") 
    flierprops = dict(marker='o', markersize=1, linestyle='none')

    plt.boxplot(boxplot_data, positions=positions, patch_artist=True,
                boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops, flierprops=flierprops)
    cleaned_method_names = [method.replace('ML_', '') for method in sorted_methods]
    plt.xticks(positions, cleaned_method_names, rotation=90)
    plt.xlabel('Method')
    plt.ylabel('Aggregated ARI Scores')
    plt.title('Aggregated ARI Scores by Method')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{save_path}GMM ARI values vs. each method (sorted).png', dpi=300)
    plt.savefig(f'{save_path}GMM ARI values vs. each method (sorted).svg', dpi=300)
    plt.clf()




