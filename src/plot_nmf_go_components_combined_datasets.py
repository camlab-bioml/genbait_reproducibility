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

def plot_nmf_go_combined(dataset_paths, component_ranges, save_path, output_prefix):
    """
    Loads NMF correlation results, performs statistical analysis, and generates a boxplot visualization.

    Args:
        dataset_paths (list): List of dataset paths containing NMF correlation results.
        component_ranges (list): List of component ranges corresponding to each dataset.
        output_prefix (str): Prefix for saving output files (e.g., "plots/nmf_correlation").
    """

    def load_and_aggregate_data(dataset_path, components_range):
        """Load NMF correlation data from a dataset and aggregate it across components."""
        ga_data = pickle.load(open(os.path.join(dataset_path, 'nmf_go_components_scores_ga.pkl'), 'rb'))
        random_data = pickle.load(open(os.path.join(dataset_path, 'nmf_go_components_scores_random.pkl'), 'rb'))
        ml_data = pickle.load(open(os.path.join(dataset_path, 'nmf_go_components_scores_ml.pkl'), 'rb'))

        aggregated_data = {}
        for name, data in [('GA', ga_data), ('Random', random_data)] + list(ml_data.items()):
            aggregated = {n: [] for n in components_range}
            for feature_data in data.values():
                for comp, values in feature_data.items():
                    if comp in components_range:
                        aggregated[comp].extend(values)
            aggregated_data[name] = aggregated
        return aggregated_data

    # Load datasets
    aggregated_data_all = [load_and_aggregate_data(dataset_paths[i], component_ranges[i]) for i in range(len(dataset_paths))]

    # Aggregate data for statistical testing
    all_methods = set().union(*[set(data.keys()) for data in aggregated_data_all])
    all_data = {method: [] for method in all_methods}

    for method in all_methods:
        for data in aggregated_data_all:
            if method in data:
                all_data[method].extend([score for comp_scores in data[method].values() for score in comp_scores])

    # Perform pairwise Mann-Whitney U tests
    method_pairs = list(itertools.combinations(all_data.keys(), 2))
    p_values = []

    for method1, method2 in method_pairs:
        stat, p_value = stats.mannwhitneyu(all_data[method1], all_data[method2], alternative='two-sided')
        p_values.append((method1, method2, p_value))

    # Apply multiple hypothesis correction (Benjamini-Hochberg)
    method_names_1, method_names_2, raw_p_values = zip(*p_values)
    adjusted_p_values = multipletests(raw_p_values, method='fdr_bh')[1]

    # Save statistical significance results
    stats_df = pd.DataFrame({'Method 1': method_names_1, 'Method 2': method_names_2, 'P-value': raw_p_values, 'Adjusted P-value': adjusted_p_values})
    stats_df.to_csv(f'{save_path}{output_prefix}_statistical_significance.csv', index=False)

    # Compute average median values for each method
    median_values = {method: np.median(scores) for method, scores in all_data.items()}

    # Save median values
    median_df = pd.DataFrame(list(median_values.items()), columns=['Method', 'Median Value'])
    median_df.to_csv(f'{save_path}{output_prefix}_median_values.csv', index=False)

    # Mapping and ordering methods for visualization
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

    ordered_methods = ['GA', 'nn', 'rf', 'gbm', 'xgb', 'lasso', 'ridge', 'elastic_net', 'mutual_info_classif', 'f_classif', 'chi_2', 'Random']
    sorted_methods = [method for method in ordered_methods if method in all_data]

    mapped_sorted_methods = [methods_name_mapping.get(method, method) for method in sorted_methods]

    # Plot visualization
    plt.figure(figsize=(12, 6))
    colors = ['#F7941D', '#009444', '#FF5733']
    positions = np.array(range(len(mapped_sorted_methods)))

    flierprops = dict(marker='o', markersize=1, markerfacecolor='black', markeredgecolor='black')

    all_data_points = []
    for i, method in enumerate(sorted_methods):
        data_per_dataset = [aggregated_data_all[d].get(method, {}).values() for d in range(len(dataset_paths))]
        for d in range(len(dataset_paths)):
            data = [score for comp_scores in data_per_dataset[d] for score in comp_scores]
            all_data_points.extend(data)
            plt.boxplot(data, positions=[positions[i] - 0.2 + (0.2 * d)], widths=0.18, patch_artist=True, boxprops=dict(facecolor=colors[d], color=colors[d]), medianprops=dict(color='black'), flierprops=flierprops)

    # Highlight the best method
    best_method = max(all_data, key=lambda k: np.median(all_data[k]))
    best_method_position = positions[sorted_methods.index(best_method)]
    plt.axvspan(best_method_position - 0.5, best_method_position + 0.5, color='#B9B9B5', alpha=0.3)

    # Adjust plot settings
    plt.xticks(positions, mapped_sorted_methods, rotation=90, fontsize=18)
    plt.ylabel("Mean NMF GO Jaccard index", fontsize=18)
    plt.legend(handles=[
        mpatches.Patch(color=colors[0], label='Dataset 1'),
        mpatches.Patch(color=colors[1], label='Dataset 2'),
        mpatches.Patch(color=colors[2], label='Dataset 3')
    ], loc='lower right', fontsize=18)
    plt.tight_layout()

    # Save updated plots
    plt.savefig(f"{save_path}{output_prefix}_comparison.png", dpi=300)
    plt.savefig(f"{save_path}{output_prefix}_comparison.svg", dpi=300)
    plt.savefig(f"{save_path}{output_prefix}_comparison.pdf", dpi=300)
    plt.clf()


