from matplotlib import patches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


# Specify the paths to the pickle files for each dataset
dataset1_file = '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake original gradient penalty/plots/go_results.pkl'
dataset2_file = '/Users/vesalkasmaeifar/vesal/PhD_Project/cell map/scripts/Bait selection/snakemake RNA Bodies gradient penalty/plots/go_results.pkl'

# Load results from both pickle files
with open(dataset1_file, 'rb') as f:
    results1 = pickle.load(f)

with open(dataset2_file, 'rb') as f:
    results2 = pickle.load(f)

flierprops = dict(marker='o', markersize=1, linestyle='none', color='black')
colors = ['lightblue', 'green']

# Determine all methods
methods = ['GA', 'Random'] + list(results1['ML'].keys())
medians = []

# Collect data for each method and compute medians
for method in methods:
    data1 = []
    data2 = []

    # Collect data for this method from both datasets
    if method in ['GA', 'Random']:
        data1 = [val for vals in results1[method].values() for val in vals]
        data2 = [val for vals in results2[method].values() for val in vals]
    else:
        data1 = [val for vals in results1['ML'][method].values() for val in vals]
        data2 = [val for vals in results2['ML'][method].values() for val in vals]

    # Compute medians
    median1 = np.median(data1) if data1 else float('nan')  # Safely compute median for dataset1
    median2 = np.median(data2) if data2 else float('nan')  # Safely compute median for dataset2
    average_median = np.nanmean([median1, median2])  # Calculate the average of the two medians
    medians.append((method, average_median))

# Sort methods by average median in decreasing order
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

# Sorting methods based on median values
methods_sorted = sorted(medians, key=lambda x: x[1], reverse=True)
sorted_methods = [m[0] for m in methods_sorted]

# Mapping method names
mapped_sorted_methods = [methods_name_mapping.get(method, method) for method in sorted_methods]

# Create a single figure to hold all boxplots
fig, ax = plt.subplots(figsize=(15, 6))

# Plot data for each method from both datasets, using sorted order
positions = np.arange(0, len(mapped_sorted_methods) * 1, 1)  

outlier_props = dict(marker='o', markersize=2)

for idx, method in enumerate(sorted_methods):
    data1 = []
    data2 = []

    # Collect data for this method from both datasets again for plotting
    if method in ['GA', 'Random']:
        data1 = [val for vals in results1[method].values() for val in vals]
        data2 = [val for vals in results2[method].values() for val in vals]
    else:
        data1 = [val for vals in results1['ML'][method].values() for val in vals]
        data2 = [val for vals in results2['ML'][method].values() for val in vals]

    data = [data1, data2]
    box_positions = [positions[idx] - 0.15, positions[idx] + 0.15]
    
    # Create boxplot
    bplot = ax.boxplot(data, positions=box_positions, widths=0.3, patch_artist=True, flierprops=outlier_props, medianprops=dict(color='black'))

    # Coloring each box
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# Customize axes and layout
ax.set_xticks(positions)
ax.set_xticklabels(mapped_sorted_methods, rotation=90, ha='center', fontsize=16)
ax.set_ylabel('GO retrieval percentage', fontsize=16)

legend_handles = [
    mpatches.Patch(color='lightblue', label='Dataset 1'),
    mpatches.Patch(color='green', label='Dataset 2')
]
plt.tight_layout()  # Adjust layout to make room for label rotation
plt.ylim(0, 100)
plt.legend(handles=legend_handles, loc='lower right', fontsize=12)
plt.savefig('plots/GO_retrieval_percentage_comparison.png', dpi=300)
plt.savefig('plots/GO_retrieval_percentage_comparison.svg', dpi=300)
plt.savefig('plots/GO_retrieval_percentage_comparison.pdf', dpi=300)
plt.clf()

