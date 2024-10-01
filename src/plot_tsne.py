# plot_tsne.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']



def plot_tsne(df_norm, ga_selected_baits, number_of_components, file_path):
    """
    Plot t-SNE for given dataframes: df_norm and df_subset (resulted from the best feature list).
    """
    df_subset = df_norm.loc[ga_selected_baits]
    mask = (df_subset != 0).any(axis=0)
    df_subset_reduced = df_subset.loc[:, mask]
    mask_removed = (df_subset == 0).all(axis=0)
    df_subset_removed = df_subset.loc[:, mask_removed]

    # Now, apply NMF to these common dataframes
    nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)

    # For the original dataframe
    scores_matrix_original = nmf.fit_transform(df_norm)
    basis_matrix_original = nmf.components_.T

    # For the subset dataframe
    scores_matrix_subset = nmf.fit_transform(df_subset)
    basis_matrix_subset = nmf.components_.T

    n_components = basis_matrix_original.shape[1]
    #     basis_matrix_original = basis_matrix_original / np.linalg.norm(basis_matrix_original, axis=0)
    #     basis_matrix_subset = basis_matrix_subset / np.linalg.norm(basis_matrix_subset, axis=0)

    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Get the column names of df_subset_reduced
    reduced_cols = df_subset_reduced.columns

    # Convert to a DataFrame and then add column names back
    basis_matrix_subset_reordered_df = pd.DataFrame(
        basis_matrix_subset_reordered, 
        columns=[i for i in range(basis_matrix_subset_reordered.shape[1])])

    # Get indices of reduced columns
    reduced_indices = [df_subset.columns.get_loc(c) for c in reduced_cols]

    # Filter the basis_matrix
    basis_matrix_subset_reordered_reduced = basis_matrix_subset_reordered_df.iloc[reduced_indices, :]
    basis_matrix_subset_reordered_reduced = basis_matrix_subset_reordered_reduced.to_numpy()

    y_original = []
    for i in range(basis_matrix_original.shape[0]):
        max_rank = np.argmax(basis_matrix_original[i,:]) 
        y_original.append(max_rank)
    y_original = np.asarray(y_original)  

    y_subset = []
    for i in range(basis_matrix_subset_reordered_reduced.shape[0]):
        max_rank = np.argmax(basis_matrix_subset_reordered_reduced[i,:]) 
        y_subset.append(max_rank)
    y_subset = np.asarray(y_subset) 
    tsne = TSNE(n_components=2, perplexity=20, metric="euclidean", random_state=42, n_iter=1000, n_jobs=-1)


     # Function to plot TSNE with labels and save to file
    def tsne_plot_components(tsne_data, labels, title, filename):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='tab20', alpha=0.6, s=10)
        # plt.colorbar(scatter)
        plt.xlabel('t-SNE 1', fontsize=22)
        plt.ylabel('t-SNE 2', fontsize=22)
        plt.title(title, fontsize=22)

        # Hiding the top and right axes
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
            # Label each cluster with the median number
        for i in np.unique(labels):
            median_coords = np.median(tsne_data[labels == i, :], axis=0)
            plt.text(median_coords[0], median_coords[1], str(i), fontweight='bold', ha='center', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'{filename}.svg', dpi=300)
        plt.savefig(f'{filename}.png', dpi=300)
        plt.savefig(f'{filename}.pdf', dpi=300)
        plt.close()


    def tsne_plot_highlight(tsne_data, colors, title, filename):
    # Define color mapping with labels for the legend
        color_map = {'orange': 'Remaining preys', 'grey': 'Lost preys'}
        
        # Create the plot
        plt.figure(figsize=(10, 8))

        # Plot each color group according to the legend mapping
        scatter_plots = []
        for color, label in color_map.items():
            # Filter indices for current color
            idx = np.where(colors == color)
            # Plot and store the scatter plot object for legend
            scatter = plt.scatter(tsne_data[idx, 0], tsne_data[idx, 1], c=color, label=label,
                                alpha=0.6, edgecolors='none', s=10)
            scatter_plots.append(scatter)
        
        # Adding labels and titles
        plt.xlabel('t-SNE 1', fontsize=22)
        plt.ylabel('t-SNE 2', fontsize=22)
        plt.title(title, fontsize=22)

        # Hiding the top and right axes
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Create a legend for the plot
        # plt.legend(handles=scatter_plots, fontsize=18, loc='upper right', bbox_to_anchor=(1.05,1.05))

        plt.tight_layout()
        plt.savefig(f'{filename}.svg', dpi=300)
        plt.savefig(f'{filename}.png', dpi=300)
        plt.savefig(f'{filename}.pdf', dpi=300)
        plt.close()



    # Generate and save original and subset TSNE plots
    tsne_output_original = tsne.fit_transform(basis_matrix_original)
    tsne_plot_components(tsne_output_original, y_original, 'Original baits', f"{file_path}/tsne_original")
    tsne_output_subset = tsne.fit_transform(basis_matrix_subset_reordered_reduced)
    tsne_plot_components(tsne_output_subset, y_subset, 'GA selected baits ', f"{file_path}/tsne_subset")

    # Prepare labels and colors for the plots
    y_original = np.argmax(basis_matrix_original, axis=1)
    removed_indices = [df_norm.columns.get_loc(col) for col in df_subset_removed.columns if col in df_norm.columns]
    colors = np.array(['orange'] * df_norm.shape[1])
    colors[removed_indices] = 'grey'    

    tsne_plot_highlight(tsne_output_original, colors, 'Highlighting remaining preys', f"{file_path}/tsne_highlighted_original")

    return basis_matrix_original, basis_matrix_subset_reordered_reduced, df_subset_reduced, y_original, y_subset

