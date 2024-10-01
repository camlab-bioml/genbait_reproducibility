import pandas as pd
import numpy as np
import os
import igraph as ig
import leidenalg
from sklearn.metrics.cluster import adjusted_rand_score
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

# Paths to the two datasets' results files
results_file1 = '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake original gradient penalty/plots/leiden_results.pkl'
results_file2 = '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake RNA Bodies gradient penalty/plots/leiden_results.pkl'

# Load results from the first dataset
with open(results_file1, 'rb') as f:
    results1 = pickle.load(f)

# Load results from the second dataset
with open(results_file2, 'rb') as f:
    results2 = pickle.load(f)

# Setup variables for method names
method_names = ['GA', 'Random'] + list(results1['ML'].keys())

# Aggregate scores for each method across both datasets
aggregated_scores1 = {}
aggregated_scores2 = {}
for method in method_names:
    if method in ['GA', 'Random']:
        scores1 = [score for cluster in results1[method].values() for score in cluster.values()]
        scores2 = [score for cluster in results2[method].values() for score in cluster.values()]
    else:
        scores1 = [score for cluster in results1['ML'][method].values() for score in cluster.values()]
        scores2 = [score for cluster in results2['ML'][method].values() for score in cluster.values()]
    aggregated_scores1[method] = [item for sublist in scores1 for item in sublist]
    aggregated_scores2[method] = [item for sublist in scores2 for item in sublist]

# Calculate median scores to sort methods
median_scores1 = {method: np.median(scores) for method, scores in aggregated_scores1.items()}
median_scores2 = {method: np.median(scores) for method, scores in aggregated_scores2.items()}
average_medians = {method: (median_scores1[method] + median_scores2[method]) / 2 for method in method_names}

# Sort methods based on average median score in descending order
sorted_methods = sorted(average_medians, key=average_medians.get, reverse=True)

# Generate boxplot data in sorted order for both datasets
boxplot_data1 = [aggregated_scores1[method] for method in sorted_methods]
boxplot_data2 = [aggregated_scores2[method] for method in sorted_methods]


# Rename methods and columns for clarity
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

# Apply mapping to sorted_methods
mapped_sorted_methods = [methods_name_mapping.get(method, method) for method in sorted_methods]

fig, ax = plt.subplots(figsize=(15, 6))
positions = np.arange(0, len(mapped_sorted_methods) * 1, 1)  

outlier_props = dict(marker='o', markersize=2)

for idx, method in enumerate(sorted_methods):
    data1 = boxplot_data1[idx]
    data2 = boxplot_data2[idx]
    # Position adjustments to align boxes side by side
    box_positions = [positions[idx] - 0.15, positions[idx] + 0.15]
    bplot1 = ax.boxplot(data1, positions=[box_positions[0]], widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor='lightblue'), medianprops=dict(color='black'), flierprops=outlier_props)
    bplot2 = ax.boxplot(data2, positions=[box_positions[1]], widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor='green'), medianprops=dict(color='black'), flierprops=outlier_props)

# Customize axes and layout
ax.set_xticks(positions)
ax.set_xticklabels([mapped_sorted_methods[idx].replace('ML_', '') for idx in range(len(mapped_sorted_methods))], rotation=90, ha='center', fontsize=16)
ax.set_ylabel('Leiden ARI score', fontsize=16)
legend_handles = [
    mpatches.Patch(color='lightblue', label='Dataset 1'),
    mpatches.Patch(color='green', label='Dataset 2')
]
plt.legend(handles=legend_handles, loc='lower right', fontsize=12)
plt.tight_layout()
plt.ylim(0,1)
# plt.show()

plt.savefig('plots/leiden_comparison.png', dpi=300)
plt.savefig('plots/leiden_comparison.svg', dpi=300)
plt.savefig('plots/leiden_comparison.pdf', dpi=300)
plt.clf()