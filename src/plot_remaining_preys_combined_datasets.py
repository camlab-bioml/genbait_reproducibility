import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.multitest import multipletests
import itertools
import matplotlib
# Configure matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


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

def plot_combined_remaining_preys(dataset_paths, component_ranges, output_prefix, save_path):
    """
    Loads remaining preys data, performs statistical analysis, and generates boxplot visualization.

    Args:
        dataset_paths (list): List of dataset directories containing pickle files.
        component_ranges (list): List of component ranges corresponding to each dataset.
        output_prefix (str): Prefix for saving output files (e.g., "plots/remaining_preys").
    """

    def load_and_aggregate_data(dataset_path, components_range):
        """Loads and aggregates remaining preys data from a dataset directory."""
        ga_data = pickle.load(open(os.path.join(dataset_path, 'remaining_preys_ga.pkl'), 'rb'))
        random_data = pickle.load(open(os.path.join(dataset_path, 'remaining_preys_random.pkl'), 'rb'))
        ml_data = pickle.load(open(os.path.join(dataset_path, 'remaining_preys_ml.pkl'), 'rb'))

        aggregated_data = {}
        for name, data in [('GA', ga_data), ('Random', random_data)] + list(ml_data.items()):
            aggregated = {n: [] for n in components_range}
            for feature_data in data.values():
                for comp, values in feature_data.items():
                    if comp in components_range:
                        aggregated[comp].extend(values)
            aggregated_data[name] = aggregated

        return aggregated_data

    # Load and aggregate data from all datasets
    aggregated_data_all = [load_and_aggregate_data(dataset_paths[i], component_ranges[i]) for i in range(len(dataset_paths))]

    # Combine data across datasets for statistical testing
    all_data = {method: [] for method in set(aggregated_data_all[0].keys()).union(*[d.keys() for d in aggregated_data_all[1:]])}
    for method in all_data.keys():
        for dataset in aggregated_data_all:
            if method in dataset:
                all_data[method].extend([score for comp_scores in dataset[method].values() for score in comp_scores])

    # Perform pairwise Mann-Whitney U tests
    method_pairs = list(itertools.combinations(all_data.keys(), 2))
    p_values = []

    for method1, method2 in method_pairs:
        stat, p_value = stats.mannwhitneyu(all_data[method1], all_data[method2], alternative='two-sided')
        p_values.append((method1, method2, p_value))

    # Apply multiple hypothesis correction (Benjamini-Hochberg)
    method_names_1, method_names_2, raw_p_values = zip(*p_values)
    adjusted_p_values = multipletests(raw_p_values, method='fdr_bh')[1]

    # Mapping method names for readability
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

    # Rename methods in statistical results
    mapped_method_names_1 = [methods_name_mapping.get(method, method) for method in method_names_1]
    mapped_method_names_2 = [methods_name_mapping.get(method, method) for method in method_names_2]

    stats_df = pd.DataFrame({
        'Method 1': mapped_method_names_1,
        'Method 2': mapped_method_names_2,
        'P-value': raw_p_values,
        'Adjusted P-value': adjusted_p_values
    })

    stats_df.to_csv(f'{save_path}{output_prefix}_statistical_significance.csv', index=False)

    # Compute average median values for each method
    median_values = {method: np.median(scores) for method, scores in all_data.items()}
    mapped_median_values = {methods_name_mapping.get(method, method): value for method, value in median_values.items()}

    # Save median values
    median_df = pd.DataFrame(list(mapped_median_values.items()), columns=['Method', 'Median Value'])
    median_df.to_csv(f'{save_path}{output_prefix}_median_values.csv', index=False)

    # Ordering methods for visualization
    ordered_methods = ['GA', 'nn', 'rf', 'gbm', 'xgb', 'lasso', 'ridge', 'elastic_net', 'mutual_info_classif', 'f_classif', 'chi_2', 'Random']
    sorted_methods = [method for method in ordered_methods if method in all_data]
    mapped_sorted_methods = [methods_name_mapping.get(method, method) for method in sorted_methods]

    # Plot visualization
    plt.figure(figsize=(12, 8))
    colors = ['#F7941D', '#009444', '#FF5733']
    positions = np.array(range(len(mapped_sorted_methods)))

    outlier_props = dict(marker='o', markersize=1)

    all_data_points = []
    for i, method in enumerate(sorted_methods):
        for j, dataset in enumerate(aggregated_data_all):
            data = []
            if method in dataset:
                data = [score * 100 for comp_scores in dataset[method].values() for score in comp_scores]
                all_data_points.extend(data)
                plt.boxplot(data, positions=[positions[i] - 0.2 + j * 0.2], patch_artist=True,
                            boxprops=dict(facecolor=colors[j], color=colors[j]),
                            widths=0.2, medianprops=dict(color='black'), flierprops=outlier_props)

    # Highlight the best method
    best_method = max(median_values, key=median_values.get)
    best_method_position = positions[sorted_methods.index(best_method)]
    plt.axvspan(best_method_position - 0.5, best_method_position + 0.5, color='#B9B9B5', alpha=0.3)

    # Adjust y-axis
    plt.tick_params(axis='y', labelsize=18)

    legend_handles = [
        mpatches.Patch(color='#F7941D', label='Dataset 1'),
        mpatches.Patch(color='#009444', label='Dataset 2'),
        mpatches.Patch(color='#FF5733', label='Dataset 3')
    ]

    plt.xticks(positions, mapped_sorted_methods, rotation=90, fontsize=18)
    plt.ylabel("Remaining preys percentage", fontsize=18)
    plt.legend(handles=legend_handles, loc='lower right', fontsize=18)
    plt.ylim(min(all_data_points), max(all_data_points))
    plt.tight_layout()

    # Save plots
    plt.savefig(f"{save_path}{output_prefix}_comparison.png", dpi=300)
    plt.savefig(f"{save_path}{output_prefix}_comparison.svg", dpi=300)
    plt.savefig(f"{save_path}{output_prefix}_comparison.pdf", dpi=300)

    plt.clf()


# Example usage
dataset_paths = [
    '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake original gradient penalty/plots',
    '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake RNA Bodies gradient penalty/plots',
    '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake nuclear Bodies gradient penalty/plots'
]

component_ranges = [range(15, 26), range(9, 20), range(10, 21)]

plot_combined_remaining_preys(dataset_paths, component_ranges,  "remaining_preys")
