import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gprofiler import GProfiler
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def load_and_aggregate_data(dataset_path, components_range):
    # Load the pickle files
    ga_data = pickle.load(open(os.path.join(dataset_path, 'nmf_scores_go_components_ga.pkl'), 'rb'))
    random_data = pickle.load(open(os.path.join(dataset_path, 'nmf_scores_go_components_random.pkl'), 'rb'))
    ml_data = pickle.load(open(os.path.join(dataset_path, 'nmf_scores_go_components_ml.pkl'), 'rb'))

    # Aggregate data
    aggregated_data = {}
    for name, data in [('GA', ga_data), ('Random', random_data)] + list(ml_data.items()):
        aggregated = {n: [] for n in components_range}
        for feature_data in data.values():
            for comp, values in feature_data.items():
                if comp in components_range:
                    aggregated[comp].extend(values)
        aggregated_data[name] = aggregated

    return aggregated_data

# Define component ranges for each dataset
components_range1 = range(15, 26)  # For dataset 1
components_range2 = range(9, 20)   # For dataset 2

# Load and aggregate data for both datasets
dataset1_path = '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake original gradient penalty/plots'
dataset2_path = '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake RNA Bodies gradient penalty/plots'

# Load and aggregate data for both datasets
aggregated_data1 = load_and_aggregate_data(dataset1_path, components_range1)
aggregated_data2 = load_and_aggregate_data(dataset2_path, components_range2)

# Calculate medians and average them
median_values = {}
for method in set(aggregated_data1.keys()).union(aggregated_data2.keys()):
    medians = []
    if method in aggregated_data1:
        data1 = [score for comp_scores in aggregated_data1[method].values() for score in comp_scores]
        if data1: medians.append(np.median(data1))
    if method in aggregated_data2:
        data2 = [score for comp_scores in aggregated_data2[method].values() for score in comp_scores]
        if data2: medians.append(np.median(data2))
    if medians:
        median_values[method] = np.mean(medians)


methods_name_mapping = {
    'GA': 'GA',
    'Random': 'Random',
    'chi_2': 'Chi-Squared',
    'f_classif': 'ANOVA F',
    'mutual_info_classif': 'Mutual Info',
    'lasso': 'Lasso',
    'ridge': 'Ridge',
    'elastic_net': 'ElasticNet',
    'rf': 'RF',
    'gbm': 'GBM',
    'xgb': 'XGB'
}

# Sort methods by their average median value
sorted_methods = sorted(median_values, key=median_values.get)[::-1]

# Mapping method names
mapped_sorted_methods = [methods_name_mapping.get(method, method) for method in sorted_methods]

# Plot settings
plt.figure(figsize=(15, 6))
colors = ['lightblue', 'green']
positions = np.array(range(len(mapped_sorted_methods)))
outlier_props = dict(marker='o', markersize=2)


for i, method in enumerate(sorted_methods):
    if method in aggregated_data1:
        data1 = [score for comp_scores in aggregated_data1[method].values() for score in comp_scores]
        plt.boxplot(data1, positions=[positions[i] - 0.15], widths=0.3, patch_artist=True, boxprops=dict(facecolor=colors[0]), medianprops=dict(color='black'), flierprops=outlier_props)
    if method in aggregated_data2:
        data2 = [score for comp_scores in aggregated_data2[method].values() for score in comp_scores]
        plt.boxplot(data2, positions=[positions[i] + 0.15], widths=0.3, patch_artist=True, boxprops=dict(facecolor=colors[1]), medianprops=dict(color='black'), flierprops=outlier_props)

legend_handles = [
    mpatches.Patch(color='lightblue', label='Dataset 1'),
    mpatches.Patch(color='green', label='Dataset 2')
]

# Customize the plot
plt.xticks(positions, mapped_sorted_methods, rotation=90, fontsize=16)
# plt.xlabel("Methods")
plt.ylabel("NMF mean GO Jaccard index score", fontsize=16)
# plt.title("Comparison of Methods across Datasets (Ordered by Average Median)")
plt.legend(handles=legend_handles, loc='lower right', fontsize=12)
plt.ylim(0, 1)
plt.tight_layout()

# Save and show the plot
save_path = 'plots'
plt.savefig(os.path.join(save_path, 'nmf_scores_go_comparison_ordered.png'), dpi=300)
plt.savefig(os.path.join(save_path, 'nmf_scores_go_comparison_ordered.svg'), dpi=300)
plt.savefig(os.path.join(save_path, 'nmf_scores_go_comparison_ordered.pdf'), dpi=300)
plt.clf()