# lost_preys_plotting.py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def plot_lost_preys_from_features(df_norm, top_features_lists, save_path='plots'):
    """
    Plot the number of lost preys for the top 10 best lists.

    Parameters:
    - top_features_lists: List of lists containing top features from GA.
    - df_norm: Normalized data used in GA.

    Returns:
    None
    """
    prey_counts = []

    for subset_features in top_features_lists:
        subset_indices = list(df_norm.index.get_indexer(subset_features))
        subset_data = df_norm.iloc[subset_indices].to_numpy()
        
        prey_count = np.count_nonzero(np.count_nonzero(subset_data, axis=0))
        prey_counts.append(prey_count)

    ranks = [f"Top {i+1}" for i in range(len(top_features_lists))]
    
    plt.figure(figsize=(10, 6))
    plt.bar(ranks, prey_counts, color='green')
    plt.xlabel('Top Lists')
    plt.ylabel('Number of Lost Preys')
    plt.title('Number of Lost Preys for Top 10 Best Lists')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}/Lost_Preys_Plot.png')
    # plt.show()

