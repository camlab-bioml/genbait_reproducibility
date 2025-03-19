import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def deep_average_scores(scores_dict):
    """
    Recursively calculates the deep average of scores across all levels of a nested dictionary.
    """
    if isinstance(scores_dict, dict):
        scores = []
        for key in scores_dict:
            result = deep_average_scores(scores_dict[key])
            if isinstance(result, list):
                scores.extend(result)
            else:
                scores.append(result)
        return np.mean(scores)
    else:
        return scores_dict

def process_file(file_path):
    """
    Processes a single pickle file, returning a dictionary of average scores for each method.
    Adjusts scores for files where lower values are better by negating the scores.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        average_scores = {}
        for method, scores in data.items():
            if method == 'ML':
                for sub_method, sub_scores in scores.items():
                    avg_score = deep_average_scores(sub_scores)
                    if 'nmf_scores_kl_all' in file_path or 'outlier_counts' in file_path:
                        avg_score = -avg_score
                    average_scores[sub_method] = avg_score
            else:
                avg_score = deep_average_scores(scores)
                if 'nmf_scores_kl_all' in file_path or 'outlier_counts' in file_path:
                    avg_score = -avg_score
                average_scores[method] = avg_score
        return average_scores

def plot_values(normalized_df, save_path):
    # Create a custom colormap with 12 discrete segments for Blues
    colormap = plt.cm.get_cmap('Blues', 12)  # 12 discrete colors
    bar_height = 0.8  # Height of the horizontal bar

    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # Background and separation aesthetics
    for i in range(0, len(normalized_df.index), 2):
        ax.axhspan(i-0.5, i+0.5, facecolor='grey', alpha=0.3)
    separation_regions = [(3, 4), (7, 8), (12, 13)]
    for start, end in separation_regions:
        ax.axvspan(start+0.5, end-0.5, facecolor='grey', alpha=0.3)

    # Plot normalized values for each column (metric), including 'Overall Score'
    for j, metric in enumerate(normalized_df.columns):
        # Rank methods within this specific metric (highest value = rank 1, lowest = rank 12)
        metric_ranks = normalized_df[metric].rank(method='dense', ascending=False)
        
        for i, method in enumerate(normalized_df.index):
            value = normalized_df.loc[method, metric]
            # Scale dot size based on value (100 for 0, 2000 for 1)
            dot_size = 100 + (value * 1000)  # Maps 0-1 to 100-2000
            text_color = 'black' if method == 'Random' else 'white'

            # Get color based on rank for this metric (rank 1 = darkest, rank 12 = lightest)
            rank = int(metric_ranks[method] - 1)  # Convert to 0-based index (0 to 11)
            # Reverse the color order: rank 1 (highest value) = darkest (11), rank 12 (lowest value) = lightest (0)
            color_idx = 11 - rank  # Invert so higher values (rank 1) get darker colors
            color = colormap(color_idx / 11)  # Normalize to 0-1 range for colormap

            if metric == 'Overall Score':
                # Plot a horizontal bar for Overall Score
                ax.barh(y=i, width=value, left=j-0.3, height=bar_height, 
                        color=color, edgecolor='black')
                # Annotate the bar with the actual value
                # plt.text(j, i, f"{value:.2f}", fontsize=14, 
                #         ha='center', va='center', color=text_color)
            else:
                # Plot a circle for other metrics
                plt.scatter(x=j, y=i, s=dot_size, c=[color], alpha=0.9, 
                          edgecolors='black', marker='o')
                # Annotate the plot with the actual value
                # plt.text(j, i, f"{value:.2f}", fontsize=14, 
                #         ha='center', va='center', color=text_color)

    # Add colorbar with discrete steps for method ranks per metric
    boundaries = np.linspace(0, 12, 13)
    norm = mcolors.BoundaryNorm(boundaries, colormap.N)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=norm), 
                       ax=ax, label='Method Rank (per Metric)', 
                       ticks=np.arange(0.5, 12.5), shrink=0.4)
    cbar.ax.set_yticklabels([f'{int(i)}' for i in range(12, 0, -1)])  # Show 12 (lightest, lowest value) to 1 (darkest, highest value)

    plt.yticks(ticks=np.arange(len(normalized_df.index)), 
              labels=normalized_df.index, fontsize=18)
    plt.xticks(ticks=np.arange(len(normalized_df.columns)), 
              labels=normalized_df.columns, rotation=90, fontsize=18)
    plt.title('Dataset 1: Go et al., 2021', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{save_path}combined_metrics_comparison_plot_values_nmf_excluded.png', dpi=300)
    plt.savefig(f'{save_path}combined_metrics_comparison_plot_values_nmf_excluded.svg', dpi=300)
    plt.savefig(f'{save_path}combined_metrics_comparison_plot_values_nmf_excluded.pdf', dpi=300)
    plt.clf()

def create_combined_metrics_plot(save_path):

    pickle_files = [f'{save_path}nmf_scores_ari_all.pkl',
                    f'{save_path}nmf_scores_cos_all.pkl', 
                    f'{save_path}nmf_scores_kl_all.pkl', 
                    f'{save_path}nmf_scores_go_components_all.pkl',
                    f'{save_path}nmf_scores_ari_all_min.pkl',
                    f'{save_path}nmf_scores_cos_all_min.pkl', 
                    f'{save_path}nmf_scores_kl_all_min.pkl', 
                    f'{save_path}nmf_scores_go_components_all_min.pkl',
                    f'{save_path}remaining_preys_all.pkl', 
                    f'{save_path}go_results.pkl', 
                    f'{save_path}gmm_results.pkl', 
                    f'{save_path}gmm_hard_results.pkl', 
                    f'{save_path}leiden_results.pkl']

    combined_df = pd.DataFrame()

    # Process each file and update the DataFrame with the average scores
    for pickle_file in pickle_files:
        file_averages = process_file(pickle_file)
        for method, avg_score in file_averages.items():
            # Simplify column name by removing path and extension
            column_name = f'{pickle_file.split("/")[-1][:-4]}_Score'
            combined_df.at[method, column_name] = avg_score

    # Rename methods and columns for clarity
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
        'nn' : 'Neural Network'
    }

    metrics_name_mapping = {
        'remaining_preys_all_Score': 'Remaining preys percentage',
        'go_results_Score': 'GO terms retrieval percentage',
        'gmm_results_Score': 'Mean GMM  Pearson correlation',
        'gmm_hard_results_Score': 'GMM ARI',
        'leiden_results_Score': 'Leiden ARI',
        'nmf_scores_ari_all_Score': 'NMF ARI',
        'nmf_scores_cos_all_Score': 'Mean NMF Cosine similarity',
        'nmf_scores_kl_all_Score': 'Mean NMF KL divergence',
        'nmf_scores_go_components_all_Score': 'Mean NMF GO Jaccard index',
        'nmf_scores_ari_all_min_Score': 'Min NMF purity score',
        'nmf_scores_cos_all_min_Score': 'Min NMF Cosine similarity',
        'nmf_scores_kl_all_min_Score': 'Max NMF KL divergence',
        'nmf_scores_go_components_all_min_Score': 'Min NMF GO Jaccard index',
        'Overall Score': 'Overall Score'
    }
    combined_df = combined_df.rename(index=methods_name_mapping)
    combined_df = combined_df.rename(columns=metrics_name_mapping)

    # Normalize the DataFrame (excluding the Overall Score and NMF mean correlation score)
    exclude_columns = ['NMF mean Pearson correlation']
    columns_for_normalization = [col for col in combined_df.columns if col not in exclude_columns]
    normalized_df = combined_df[columns_for_normalization].fillna(0).apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    # Calculate and normalize the overall score for each method, excluding the NMF mean correlation score
    normalized_df['Overall Score'] = normalized_df.mean(axis=1)
    normalized_df['Overall Score'] = (normalized_df['Overall Score'] - normalized_df['Overall Score'].min()) / (normalized_df['Overall Score'].max() - normalized_df['Overall Score'].min())
    normalized_df.to_csv(f'{save_path}all_metrics.csv')

    # Custom sorting to ensure 'GA' and 'Random' are always at the top
    ga_df = normalized_df.loc[['GENBAIT']]
    random_df = normalized_df.loc[['Random']]
    rest_df = normalized_df.drop(['GENBAIT', 'Random'])
    rest_df_sorted = rest_df.sort_values(by='Overall Score', ascending=False)
    sorted_normalized_df = pd.concat([rest_df_sorted, random_df, ga_df])
    sorted_normalized_df = sorted_normalized_df.sort_values(by='Overall Score', ascending=True)

    plot_values(sorted_normalized_df, save_path)


