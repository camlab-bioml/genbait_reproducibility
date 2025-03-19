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

def plot_combined_go(dataset_files, save_path, output_prefix):
    """
    Loads GO retrieval data, performs statistical analysis, and generates boxplot visualization.

    Args:
        dataset_files (list): List of dataset file paths containing GO retrieval results.
        output_prefix (str): Prefix for saving output files (e.g., "plots/go_retrieval").
    """

    # Load results from all datasets
    results_all = [pickle.load(open(file, 'rb')) for file in dataset_files]

    flierprops = dict(marker='o', markersize=1, linestyle='none', color='black')
    colors = ['#F7941D', '#009444', '#FF5733']

    # Determine all methods
    methods = ['GA', 'Random'] + list(results_all[0]['ML'].keys())
    medians = []
    aggregated_data = {}

    # Collect data for each method and compute medians
    for method in methods:
        data_sets = []
        for results in results_all:
            if method in ['GA', 'Random']:
                data = [val for vals in results[method].values() for val in vals]
            else:
                data = [val for vals in results['ML'][method].values() for val in vals]
            data_sets.append(data)
        
        # Store aggregated data
        aggregated_data[method] = [val for dataset in data_sets for val in dataset]

        # Compute medians
        medians.append((method, np.nanmean([np.median(data) if data else float('nan') for data in data_sets])))

    # Sort median values
    sorted_median_values = sorted(medians, key=lambda x: x[1], reverse=True)
    median_values_df = pd.DataFrame(sorted_median_values, columns=['Method', 'Median Value'])
    median_values_df.to_csv(f'{save_path}{output_prefix}_median_values.csv', index=False)

    # Perform pairwise Mann-Whitney U tests
    method_pairs = list(itertools.combinations(aggregated_data.keys(), 2))
    p_values = []

    for method1, method2 in method_pairs:
        if aggregated_data[method1] and aggregated_data[method2]:  # Ensure lists are non-empty
            stat, p_value = stats.mannwhitneyu(aggregated_data[method1], aggregated_data[method2], alternative='two-sided')
        else:
            p_value = np.nan  # Assign NaN if one of the lists is empty
        p_values.append((method1, method2, p_value))

    # Apply multiple hypothesis correction (Benjamini-Hochberg)
    method_names_1, method_names_2, raw_p_values = zip(*p_values)
    adjusted_p_values = multipletests(raw_p_values, method='fdr_bh')[1]

    # Save statistical significance results
    stats_df = pd.DataFrame({'Method 1': method_names_1, 'Method 2': method_names_2, 'P-value': raw_p_values, 'Adjusted P-value': adjusted_p_values})
    stats_df.to_csv(f'{save_path}{output_prefix}_statistical_significance.csv', index=False)

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

    # Order methods by the specified order
    ordered_methods = ['GA', 'nn', 'rf', 'gbm', 'xgb', 'lasso', 'ridge', 'elastic_net', 'mutual_info_classif', 'f_classif', 'chi_2', 'Random']
    sorted_methods = [method for method in ordered_methods if method in [m[0] for m in medians]]
    mapped_sorted_methods = [methods_name_mapping.get(method, method) for method in sorted_methods]

    # Plot visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    positions = np.arange(0, len(mapped_sorted_methods) * 1, 1)

    all_data = []
    for idx, method in enumerate(sorted_methods):
        data_sets = []
        for results in results_all:
            if method in ['GA', 'Random']:
                data = [val for vals in results[method].values() for val in vals]
            else:
                data = [val for vals in results['ML'][method].values() for val in vals]
            data_sets.append(data)
            all_data.extend(data)

        box_positions = [positions[idx] - 0.2, positions[idx], positions[idx] + 0.2]

        bplot = ax.boxplot(data_sets, positions=box_positions, widths=0.2, patch_artist=True, flierprops=flierprops, medianprops=dict(color='black'))

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)

    # Determine the method with the highest average median
    best_method = max(medians, key=lambda x: x[1])[0]
    best_method_position = positions[sorted_methods.index(best_method)]

    # Shade the background for the best method
    plt.axvspan(best_method_position - 0.5, best_method_position + 0.5, color='#B9B9B5', alpha=0.3)

    ax.set_xticks(positions)
    ax.set_xticklabels(mapped_sorted_methods, rotation=90, ha='center', fontsize=18)
    ax.set_ylabel('GO retrieval percentage', fontsize=18)

    # Increase y-axis tick font size
    ax.tick_params(axis='y', labelsize=18)

    legend_handles = [
        mpatches.Patch(color='#F7941D', label='Dataset 1'),
        mpatches.Patch(color='#009444', label='Dataset 2'),
        mpatches.Patch(color='#FF5733', label='Dataset 3')
    ]
    plt.tight_layout()
    plt.ylim(min(all_data), max(all_data))
    plt.legend(handles=legend_handles, loc='lower right', fontsize=18)

    # Save plots
    plt.savefig(f'{save_path}{output_prefix}_comparison.png', dpi=300)
    plt.savefig(f'{save_path}{output_prefix}_comparison.svg', dpi=300)
    plt.savefig(f'{save_path}{output_prefix}_comparison.pdf', dpi=300)
    plt.clf()

