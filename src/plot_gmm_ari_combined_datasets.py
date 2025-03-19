import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.multitest import multipletests
import itertools
import pickle
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def plot_gmm_ari_combined(dataset_files, save_path, output_prefix):
    """
    Loads GMM ARI results, performs statistical analysis, and generates boxplot visualization.

    Args:
        dataset_files (list): List of dataset file paths containing GMM ARI results.
        output_prefix (str): Prefix for saving output files (e.g., "plots/gmm_hard").
    """

    # Load results from all datasets
    results_all = [pickle.load(open(file, 'rb')) for file in dataset_files]

    # Determine all methods
    methods = ['GA', 'Random'] + list(results_all[0]['ML'].keys())

    # Aggregate scores for each method across datasets
    aggregated_scores_all = {method: [] for method in methods}
    aggregated_scores_per_dataset = [{} for _ in dataset_files]

    for i, results in enumerate(results_all):
        for method in methods:
            if method in ['GA', 'Random']:
                scores = [score for cluster in results[method].values() for score in cluster.values()]
            else:
                scores = [score for cluster in results['ML'][method].values() for score in cluster.values()]

            aggregated_scores_all[method].extend(scores)
            aggregated_scores_per_dataset[i][method] = [item for sublist in scores for item in sublist]

    # Compute median scores
    median_scores = {method: [np.median(aggregated_scores_per_dataset[i].get(method, [])) for i in range(len(dataset_files))] for method in methods}
    average_medians = {method: np.nanmean(scores) for method, scores in median_scores.items()}

    # Perform pairwise Mann-Whitney U tests
    method_pairs = list(itertools.combinations(aggregated_scores_all.keys(), 2))
    p_values = []

    for method1, method2 in method_pairs:
        if aggregated_scores_all[method1] and aggregated_scores_all[method2]:  # Ensure lists are non-empty
            _, p_value = stats.mannwhitneyu(aggregated_scores_all[method1], aggregated_scores_all[method2], alternative='two-sided')
        else:
            p_value = np.nan  # Assign NaN if one of the lists is empty
        p_values.append((method1, method2, p_value))

    # Apply multiple hypothesis correction (Benjamini-Hochberg)
    method_names_1, method_names_2, raw_p_values = zip(*p_values)
    adjusted_p_values = multipletests(raw_p_values, method='fdr_bh')[1]

    # Save statistical significance results
    stats_df = pd.DataFrame({'Method 1': method_names_1, 'Method 2': method_names_2, 'P-value': raw_p_values, 'Adjusted P-value': adjusted_p_values})
    stats_df.to_csv(f'{save_path}{output_prefix}_statistical_significance.csv', index=False)

    # Order methods
    ordered_methods = ['GA', 'nn', 'rf', 'gbm', 'xgb', 'lasso', 'ridge', 'elastic_net', 'mutual_info_classif', 'f_classif', 'chi_2', 'Random']
    sorted_methods = [method for method in ordered_methods if method in average_medians]

    # Save median values
    median_values_df = pd.DataFrame([(method, average_medians[method]) for method in sorted_methods], columns=['Method', 'Median Value'])
    median_values_df.to_csv(f'{save_path}{output_prefix}_median_values.csv', index=False)

    # Mapping method names for clarity
    methods_name_mapping = {
        'GA': 'GENBAIT',
        'Random': 'Random',
        'chi_2': 'Chi-Squared',
        'f_classif': 'ANOVA F',
        'mutual_info_classif': 'Mutual Info',
        'lasso': 'Lasso',
        'ridge': 'Ridge',
        'elastic_net': 'ElasticNet',
        'rf': 'RF',
        'gbm': 'GBM',
        'xgb': 'XGB',
        'nn': 'Neural Network'
    }

    # Apply mapping to sorted_methods
    mapped_sorted_methods = [methods_name_mapping.get(method, method) for method in sorted_methods]

    # Plot visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    positions = np.arange(len(mapped_sorted_methods))

    colors = ['#F7941D', '#009444', '#FF5733']
    flierprops = dict(marker='o', markersize=1)

    all_data = []
    for idx, method in enumerate(sorted_methods):
        box_data = [aggregated_scores_per_dataset[i].get(method, []) for i in range(len(dataset_files))]
        all_data.extend([val for dataset in box_data for val in dataset])

        box_positions = [positions[idx] - 0.2, positions[idx], positions[idx] + 0.2]
        for i in range(len(dataset_files)):
            ax.boxplot(box_data[i], positions=[box_positions[i]], widths=0.18, patch_artist=True,
                       boxprops=dict(facecolor=colors[i], color=colors[i]), medianprops=dict(color='black'), flierprops=flierprops)

    # Determine the method with the highest average median
    best_method = max(average_medians, key=average_medians.get)
    best_method_position = sorted_methods.index(best_method)

    # Highlight the best method
    plt.axvspan(best_method_position - 0.5, best_method_position + 0.5, color='#B9B9B5', alpha=0.3)

    ax.set_xticks(positions)
    ax.set_xticklabels(mapped_sorted_methods, rotation=90, ha='center', fontsize=18)
    ax.set_ylabel('GMM ARI score', fontsize=18)

    plt.legend(handles=[
        mpatches.Patch(color=colors[0], label='Dataset 1'),
        mpatches.Patch(color=colors[1], label='Dataset 2'),
        mpatches.Patch(color=colors[2], label='Dataset 3')
    ], loc='lower right', fontsize=18)

    plt.tight_layout()
    plt.ylim(min(all_data), max(all_data))

    # Save plots
    plt.savefig(f'{save_path}{output_prefix}_comparison.png', dpi=300)
    plt.savefig(f'{save_path}{output_prefix}_comparison.svg', dpi=300)
    plt.savefig(f'{save_path}{output_prefix}_comparison.pdf', dpi=300)
    plt.clf()


