# correlation_plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def plot_components_correlations_reordered(df_norm, selected_baits, number_of_components, method, file_path):
    """
    Plots the correlations between components of the original and subset data.

    Parameters:
    - df_norm: DataFrame containing the original data.
    - selected_baits: List of selected baits.

    Returns:
    None
    """
    
    # Convert dataframes to numpy arrays
    original_data = df_norm.to_numpy()
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    subset_data = original_data[subset_indices, :]

    # Decomposition with NMF
    nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_original = nmf.fit_transform(original_data)
    basis_matrix_original = nmf.components_.T

    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Calculating cosine similarity and reordering the basis matrix of the subset
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    _, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Calculating correlation matrix
    corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:number_of_components, number_of_components:]

   # Plotting
    plt.figure(figsize=(10, 8))  # Adjusted size to more typical aspect ratio
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="seismic", vmin=-1, vmax=1, square=True, cbar=False, annot_kws={"size": 10})
    cbar = ax.figure.colorbar(ax.collections[0], ax=ax, location="right", shrink=0.2, aspect=10)
    cbar.set_label('Correlation Coefficient') 
    plt.xlabel("Components of Original Basis Matrix", fontsize=14)
    plt.ylabel("Components of Subset Basis Matrix (reordered)", fontsize=14)
    plt.tight_layout()  # Moved before savefig
    plt.title(f'{method} subset')
    plt.savefig(f'{file_path}{method}_component_corr_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{file_path}{method}_component_corr_plot.svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{file_path}{method}_component_corr_plot.pdf', dpi=300, bbox_inches='tight')
    plt.clf()

