import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def plot_preys_for_baits(saint_file, primary_baits, primary_baits_file, selected_baits_file,
                        df_norm_file, n_components, file_path):
    # Load datasets
    df = pd.read_csv(saint_file, sep='\t')
    df = df[df['BFDR'] <= 0.01]
    if primary_baits == True:
        primary_baits = pd.read_csv(primary_baits_file, index_col=0)
        primary_baits = list(primary_baits['Gene Symbols'])
        df = df[df['Bait'].isin(primary_baits)]

    selected_baits_df = pd.read_csv(selected_baits_file)
    selected_baits = selected_baits_df['selected_baits']

    bait_preys_dict = {}
    for bait in selected_baits:
        preys = df[df['Bait'] == bait]['PreyGene'].to_list()
        bait_preys_dict[bait] = preys

    df_norm = pd.read_csv(df_norm_file, index_col=0)

    np.random.seed(42)
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=42)
    nmf.fit(df_norm)

    tsne = TSNE(n_components=2, perplexity=20, metric="euclidean", random_state=42, n_iter=1000, n_jobs=-1)
    tsne_results = tsne.fit_transform(df_norm.T)

    tsne_df = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    tsne_df['protein'] = df_norm.columns
    tsne_df['color'] = 'default'
    for bait, protein_list in bait_preys_dict.items():
        tsne_df.loc[tsne_df['protein'].isin(protein_list), 'color'] = 'highlighted'

    num_plots = len(bait_preys_dict)
    num_cols = min(num_plots, 5)
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 30))
    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    axes = axes.flatten()

    for i, (bait, protein_list) in enumerate(bait_preys_dict.items()):
        tsne_subset = tsne_df[tsne_df['protein'].isin(protein_list)].copy()
        tsne_subset['color'] = 'highlighted'
        tsne_plot_df = tsne_df.copy()
        tsne_plot_df['color'] = 'default'
        tsne_plot_df.loc[tsne_plot_df['protein'].isin(protein_list), 'color'] = 'highlighted'
        tsne_plot_df = tsne_plot_df.sort_values('color')

        sns.scatterplot(
            x='tsne1', y='tsne2', hue='color',
            palette={'default': 'lightgray', 'highlighted': 'red'},
            data=tsne_plot_df, legend=None, alpha=0.7, ax=axes[i], s=3
        )
        axes[i].set_title(f'Proteins in List: {bait}', fontsize=10)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('t-SNE Plot of all selected baits', fontsize=20)
    plt.savefig(f'{file_path}preys_for_each_bait.png', dpi=400)

