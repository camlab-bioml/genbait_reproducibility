import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


# method_names = ['GA', 'Random', 'chi_2', 'f_classif', 'mutual_info_classif', 'lasso', 'ridge', 'elastic_net', 'rf', 'gbm', 'xgb']
ml_names = ['chi_2', 'f_classif', 'mutual_info_classif', 'lasso', 'ridge', 'elastic_net', 'rf', 'gbm', 'xgb']

def process_ga_results(df_norm, components_range, ga_path):
    """
    Processes each directory, grouping files by the number of features, and aggregates
    the diagonal values of the correlation matrices for each number of components.
    """
    results_ga = {}
    directory = ga_path
    # Group files by number of features
    feature_groups = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            parts = file_name.split('_')
            num_features = int(parts[3])  # Extracting the number of features
            if num_features not in feature_groups:
                feature_groups[num_features] = []
            feature_groups[num_features].append(file_name)

    # Process each group of features
    for num_features, files in feature_groups.items():
        results_ga[num_features] = {n: [] for n in components_range}
        for file_name in files:
            selected_baits_df = pd.read_csv(os.path.join(directory, file_name))
            selected_baits = selected_baits_df.iloc[:, 0].tolist()

            for number_of_components in components_range:
                original_data = df_norm.to_numpy()
                subset_indices = list(df_norm.index.get_indexer(selected_baits))
                if len(subset_indices) > number_of_components:
                            subset_data = original_data[subset_indices, :]

                            nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
                            scores_matrix_original = nmf.fit_transform(original_data)
                            basis_matrix_original = nmf.components_.T

                            scores_matrix_subset = nmf.fit_transform(subset_data)
                            basis_matrix_subset = nmf.components_.T

                            cosine_similarity1 = np.dot(basis_matrix_original.T, basis_matrix_subset)
                            cost_matrix = 1 - cosine_similarity1
                            row_ind, col_ind = linear_sum_assignment(cost_matrix)
                            basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]
                            cos_sim_matrix = cosine_similarity(basis_matrix_original.T, basis_matrix_subset_reordered.T)
                            results_ga[num_features][number_of_components].append(np.mean(np.diag(cos_sim_matrix)))
    
    return results_ga

def process_random_baseline(df_norm, components_range, random_path):
    """
    Processes the random baseline considering the specific structure of the CSV file.
    """
    df_random_baits_all = pd.read_csv(f'{random_path}all_random_baits.csv', header=None)
    results_random = {length: {n: [] for n in components_range} for length in range(30, 81)}
    original_data = df_norm.to_numpy()
    for bait_length in range(30, 81):
        column_start = (bait_length - 30) * 10
        column_end = column_start + 10

        for number_of_components in components_range:
            
            for column in range(column_start, column_end):
                random_selected_baits = df_random_baits_all[column].dropna().tolist()
                subset_data = df_norm.loc[random_selected_baits,:].to_numpy()
                
                if len(subset_data) >= number_of_components:
                    nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
                    scores_matrix_original = nmf.fit_transform(original_data)
                    basis_matrix_original = nmf.components_.T
                    scores_matrix_subset = nmf.fit_transform(subset_data)
                    basis_matrix_subset = nmf.components_.T
                    cosine_similarity1 = np.dot(basis_matrix_original.T, basis_matrix_subset)
                    cost_matrix = 1 - cosine_similarity1
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]
                    cos_sim_matrix = cosine_similarity(basis_matrix_original.T, basis_matrix_subset_reordered.T)
                    results_random[bait_length][number_of_components].append(np.mean(np.diag(cos_sim_matrix)))
    
    return results_random

def process_ml_results(df_norm, components_range, ml_path):
    """
    Processes the ML_results directory for various ML methods.
    """
    results_ml = {method: {entity_count: {n: [] for n in components_range} for entity_count in range(30, 81)} for method in ml_names}
    original_data = df_norm.to_numpy()
    for file_name in os.listdir(ml_path):
        if file_name.endswith('.csv'):
            methods_df = pd.read_csv(os.path.join(ml_path, file_name))

            for method in ml_names:
                for entity_count in range(30, 81):
                    if method in methods_df:
                        selected_baits = methods_df[method].dropna().iloc[:entity_count].tolist()
                        subset_indices = list(df_norm.index.get_indexer(selected_baits))
                        if len(subset_indices) == 0:
                            continue
                        subset_data = df_norm.to_numpy()[subset_indices, :]

                        for number_of_components in components_range:
                            if len(subset_indices) >= number_of_components:
                                nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
                                scores_matrix_original = nmf.fit_transform(original_data)
                                basis_matrix_original = nmf.components_.T

                                scores_matrix_subset = nmf.fit_transform(subset_data)
                                basis_matrix_subset = nmf.components_.T

                                cosine_similarity1 = np.dot(basis_matrix_original.T, basis_matrix_subset)
                                cost_matrix = 1 - cosine_similarity1
                                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                                basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]
                                cos_sim_matrix = cosine_similarity(basis_matrix_original.T, basis_matrix_subset_reordered.T)
                                results_ml[method][entity_count][number_of_components].append(np.mean(np.diag(cos_sim_matrix)))
    
    return results_ml


def plot_nmf_cos_scores(df_norm, components_range, ga_path, ml_path, random_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    ga_pickle_path = os.path.join(save_path, 'nmf_scores_cos_ga.pkl')
    random_pickle_path = os.path.join(save_path, 'nmf_scores_cos_random.pkl')
    ml_pickle_path = os.path.join(save_path, 'nmf_scores_cos_ml.pkl')

    # Check if the pickle files exist
    if not all([os.path.exists(ga_pickle_path), os.path.exists(random_pickle_path), os.path.exists(ml_pickle_path)]):
        # If any pickle file doesn't exist, process the results again

        ga_results = process_ga_results(df_norm, components_range, ga_path)
        random_results = process_random_baseline(df_norm, components_range, random_path)
        ml_results = process_ml_results(df_norm, components_range, ml_path)

        # Save the processed results
        with open(ga_pickle_path, 'wb') as f:
            pickle.dump(ga_results, f)
        with open(random_pickle_path, 'wb') as f:
            pickle.dump(random_results, f)
        with open(ml_pickle_path, 'wb') as f:
            pickle.dump(ml_results, f)
    

    
    ga_data = pickle.load(open(os.path.join(save_path, 'nmf_scores_cos_ga.pkl'), 'rb'))
    random_data = pickle.load(open(os.path.join(save_path, 'nmf_scores_cos_random.pkl'), 'rb'))
    ml_data = pickle.load(open(os.path.join(save_path, 'nmf_scores_cos_ml.pkl'), 'rb'))

    colormap = plt.cm.Blues  # Change to any colormap you prefer
    color_indices = np.linspace(0, 1, len(components_range))
    component_colors = [colormap(i) for i in color_indices]  # Generate colors from the colormap
    flierprops = dict(marker='o', markersize=1, linestyle='none')

    plt.figure(figsize=(20, 8))
    # Load the pickle files
    # Aggregate data for GA and Random
    aggregated_ga = {n: [] for n in components_range}
    for bait_values in ga_data.values():
        for comp, values in bait_values.items():
            aggregated_ga[comp].extend(values)
    
    aggregated_random = {n: [] for n in components_range}
    for bait_values in random_data.values():
        for comp, values in bait_values.items():
            aggregated_random[comp].extend(values)
    
    # Aggregate data for ML methods
    aggregated_ml_methods = {}
    for method in ml_data:
        aggregated_ml = {n: [] for n in components_range}
        for feature_data in ml_data[method].values():
            for comp, values in feature_data.items():
                aggregated_ml[comp].extend(values)
        aggregated_ml_methods[method] = aggregated_ml

    # Initialize dictionary to hold all aggregated data for plotting
    all_data = {'GA': aggregated_ga, 'Random': aggregated_random}
    all_data.update(aggregated_ml_methods)

    # Define the position of the box plots
    method_names = ['GA', 'Random'] + list(aggregated_ml_methods.keys())
    positions = np.arange(len(method_names))
    # component_colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'red'][:len(components_range)]
    patch_list = [mpatches.Patch(color=component_colors[i], label=f'Components: {n}') for i, n in enumerate(components_range)]
    box_plot_width = 2

    spacing_factor = 3  # Adjust this factor to increase/decrease the spacing
    positions = [i * spacing_factor * len(components_range) for i in range(len(method_names))]

    # Plot the data
    for idx, method in enumerate(method_names):
        # Calculate the position offset for the box plots of each component within a method group
        for comp_idx, comp in enumerate(components_range):
            # Adjust the position calculation to accommodate the new spacing
            pos = positions[idx] + comp_idx * box_plot_width - (len(components_range) * box_plot_width / 2)
            plt.boxplot(all_data[method][comp], positions=[pos], widths=box_plot_width, patch_artist=True, boxprops=dict(facecolor=component_colors[comp_idx]), flierprops=flierprops)

    plt.xticks(positions, method_names, rotation=90)
    plt.xlabel("Methods")
    plt.ylabel("NMF Cos Diagonal Mean Values")
    plt.title("Comparison of Methods over Different Number of Components")
    plt.legend(handles=patch_list, bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'nmf cos scores vs. each method.png'), dpi=300)
    plt.savefig(os.path.join(save_path, 'nmf cos scores vs. each method.svg'), dpi=300)
    plt.clf()
    # Summary boxplot with methods sorted by median NMF score
    summary_data = {}
    medians = {}

    # Aggregate scores for each method and calculate medians
    for method in method_names:
        all_scores = [score for comp_scores in all_data[method].values() for score in comp_scores]
        summary_data[method] = all_scores
        medians[method] = np.median(all_scores)
    
    # Sort methods by median value
    sorted_methods = sorted(method_names, key=lambda x: medians[x], reverse=True)

    spacing_factor = 0.5
    plt.figure(figsize=(12, 9))
    for i, method in enumerate(sorted_methods):
        position = spacing_factor * i
        plt.boxplot(summary_data[method], positions=[position], patch_artist=True, boxprops=dict(facecolor='lightblue'), flierprops=flierprops)

    plt.xticks([i * spacing_factor for i in range(len(sorted_methods))], sorted_methods, rotation=90)
    plt.xlabel("Methods")
    plt.ylabel("Aggregated NMF Cos Diagonal Mean Values")
    plt.title("Aggregated Comparison of Methods")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'nmf cos scores vs. each method (sorted).png'), dpi=300)
    plt.savefig(os.path.join(save_path, 'nmf cos scores vs. each method (sorted).svg'), dpi=300)
    plt.clf()

    all_values = []
    for data in [ga_data, random_data] + list(ml_data.values()):
        for bait_values in data.values():
            for comp_values in bait_values.values():
                all_values.extend(comp_values)
    global_min = min(all_values)
    global_max = max(all_values)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(30, 20), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Plot data for GA and Random
    for idx, method in enumerate(method_names):
        ax = axes[idx]
        method_data = ga_data if method == 'GA' else random_data if method == 'Random' else ml_data[method]

        boxplot_data = []
        bait_numbers = sorted(method_data)
        for bait_num in bait_numbers:
            bait_data = [values for comp in components_range for values in method_data[bait_num].get(comp, [])]
            boxplot_data.append(bait_data)
        
        ax.boxplot(boxplot_data, positions=range(len(bait_numbers)))
        ax.set_title(method)
        ax.set_xticks(range(len(bait_numbers)))
        ax.set_xticklabels(bait_numbers)
        ax.set_ylim(global_min, global_max)  # Set the same Y-axis range for all subplots


    # Hide any unused axes
    for i in range(len(method_names), len(axes)):
        axes[i].axis('off')

    # Adjust layout and save the figure
    plt.suptitle('NMF Cos Values by Number of Baits for Each Method')
    plt.savefig(os.path.join(save_path, 'nmf cos score vs. number of baits.png'), dpi=300)
    plt.savefig(os.path.join(save_path, 'nmf cos score vs. number of baits.svg'), dpi=300)
    plt.close()


    # Define methods and initialize storage for aggregated data
    methods = ['GA', 'Random'] + list(ml_data.keys())
    all_data = {method: {n: [] for n in components_range} for method in methods}
        # Initialize figure
    fig, ax = plt.subplots(figsize=(15, 9))

    # Define a color map for the methods
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))

    # Aggregate all values for setting global y-axis limits
    all_values = []
    for data in [ga_data, random_data] + list(ml_data.values()):
        for bait_values in data.values():
            for comp_values in bait_values.values():
                all_values.extend(comp_values)
    global_min, global_max = min(all_values), max(all_values)

    # Function to aggregate scores across components for a given method's data
    def aggregate_scores(data):
        aggregated_scores = {}
        for bait_num, comp_values in data.items():
            all_scores = []
            for scores in comp_values.values():
                all_scores.extend(scores)
            aggregated_scores[bait_num] = all_scores
        return aggregated_scores

    # Prepare data for plotting
    plot_data = {
        'GA': aggregate_scores(ga_data),
        'Random': aggregate_scores(random_data)
    }
    plot_data.update({method: aggregate_scores(ml_data[method]) for method in ml_data})

    # Plot each method
    for method, color in zip(plot_data, colors):
        bait_numbers = sorted(plot_data[method])
        means = [np.mean(plot_data[method][bait]) for bait in bait_numbers]
        std_devs = [np.std(plot_data[method][bait]) for bait in bait_numbers]

        ax.plot(bait_numbers, means, label=method, color=color)
        ax.fill_between(bait_numbers, [m - s for m, s in zip(means, std_devs)], [m + s for m, s in zip(means, std_devs)], color=color, alpha=0.2)

    # Set plot parameters
    ax.set_xlabel('Number of Baits')
    ax.set_ylabel('Aggregated NMF Cos Scores')
    ax.set_title('Aggregated NMF Cos Scores by Number of Baits for Each Method')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    ax.set_xlim(left=min(bait_numbers), right=max(bait_numbers))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'nmf cos score vs. number of baits_mean with shades.png'), dpi=300)
    plt.savefig(os.path.join(save_path, 'nmf cos score vs. number of baits_mean with shades.svg'), dpi=300)
    plt.clf()
