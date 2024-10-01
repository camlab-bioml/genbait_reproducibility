import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def plot_component_overlap(labels_original, labels_subset):
    """
    Plots the overlap and non-overlap of component assignments between the original and subset data.

    Parameters:
    - labels_original: The labels for each data point in the original data.
    - labels_subset: The labels for each data point in the subset data.

    Returns:
    None
    """
    unique_labels = np.unique(labels_original)
    overlap_counts = []
    non_overlap_counts = []
    
    for label in unique_labels:
        original_indices = np.where(labels_original == label)[0]
        subset_overlap = len([i for i in original_indices if labels_subset[i] == label])
        
        overlap_counts.append(subset_overlap)
        non_overlap_counts.append(len(original_indices) - subset_overlap)
    
    # Create a stacked bar chart
    plt.figure(figsize=(12, 8))
    p1 = plt.bar(unique_labels, overlap_counts, color='blue', label='Overlap with Subset')
    p2 = plt.bar(unique_labels, non_overlap_counts, bottom=overlap_counts, color='gray', label='Not in Subset')
    
    plt.xlabel('Component Labels')
    plt.ylabel('Number of Preys')
    plt.title('Overlap of Component Assignments between Original and Subset Data')
    plt.legend()
    plt.tight_layout()
    # plt.show()

def plot_components_comparison(df_norm, selected_baits, number_of_components):
    """
    Plots the comparison of component assignments between the original and subset data.

    Parameters:
    - df_norm: DataFrame containing the original data.
    - selected_baits: List of selected baits.
    - number_of_components: The number of NMF components.

    Returns:
    None
    """
    
    # Convert dataframes to numpy arrays
    original_data = df_norm.to_numpy()
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    subset_data = original_data[subset_indices, :]

    # Decomposition with NMF
    nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
    nmf.fit(original_data)
    basis_matrix_original = nmf.components_.T
    
    nmf.fit(subset_data)
    basis_matrix_subset = nmf.components_.T

    #Calculating cosine similarity and reordering the basis matrix of the subset
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    _, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Assign labels based on the component with the highest value in the basis matrix
    labels_original_primary = np.argmax(basis_matrix_original, axis=1)
    labels_subset_primary = np.argmax(basis_matrix_subset_reordered, axis=1)
    
    plot_component_overlap(labels_original_primary, labels_subset_primary)


