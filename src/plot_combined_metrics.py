import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def combine_pkl_files(save_path):
    with open(f'{save_path}nmf_scores_ga.pkl', 'rb') as f:
        ga = pickle.load(f)
    with open(f'{save_path}nmf_scores_random.pkl', 'rb') as f:
        random = pickle.load(f)
    with open(f'{save_path}nmf_scores_ml.pkl', 'rb') as f:
        ml = pickle.load(f)
    all_methods = {}
    all_methods['GA'] = ga
    all_methods['Random'] = random
    all_methods['ML'] = ml
    with open(f'{save_path}nmf_scores_all.pkl', 'wb') as f:
        pickle.dump(all_methods, f)

    with open(f'{save_path}nmf_scores_ari_ga.pkl', 'rb') as f:
        ga = pickle.load(f)
    with open(f'{save_path}nmf_scores_ari_random.pkl', 'rb') as f:
        random = pickle.load(f)
    with open(f'{save_path}nmf_scores_ari_ml.pkl', 'rb') as f:
        ml = pickle.load(f)
    all_methods = {}
    all_methods['GA'] = ga
    all_methods['Random'] = random
    all_methods['ML'] = ml
    with open(f'{save_path}nmf_scores_ari_all.pkl', 'wb') as f:
        pickle.dump(all_methods, f)

    with open(f'{save_path}nmf_scores_cos_ga.pkl', 'rb') as f:
        ga = pickle.load(f)
    with open(f'{save_path}nmf_scores_cos_random.pkl', 'rb') as f:
        random = pickle.load(f)
    with open(f'{save_path}nmf_scores_cos_ml.pkl', 'rb') as f:
        ml = pickle.load(f)
    all_methods = {}
    all_methods['GA'] = ga
    all_methods['Random'] = random
    all_methods['ML'] = ml
    with open(f'{save_path}nmf_scores_cos_all.pkl', 'wb') as f:
        pickle.dump(all_methods, f)

    with open(f'{save_path}nmf_scores_kl_ga.pkl', 'rb') as f:
        ga = pickle.load(f)
    with open(f'{save_path}nmf_scores_kl_random.pkl', 'rb') as f:
        random = pickle.load(f)
    with open(f'{save_path}nmf_scores_kl_ml.pkl', 'rb') as f:
        ml = pickle.load(f)
    all_methods = {}
    all_methods['GA'] = ga
    all_methods['Random'] = random
    all_methods['ML'] = ml
    with open(f'{save_path}nmf_scores_kl_all.pkl', 'wb') as f:
        pickle.dump(all_methods, f)

    with open(f'{save_path}nmf_scores_go_components_ga.pkl', 'rb') as f:
        ga = pickle.load(f)
    with open(f'{save_path}nmf_scores_go_components_random.pkl', 'rb') as f:
        random = pickle.load(f)
    with open(f'{save_path}nmf_scores_go_components_ml.pkl', 'rb') as f:
        ml = pickle.load(f)
    all_methods = {}
    all_methods['GA'] = ga
    all_methods['Random'] = random
    all_methods['ML'] = ml
    with open(f'{save_path}nmf_scores_go_components_all.pkl', 'wb') as f:
        pickle.dump(all_methods, f)

    with open(f'{save_path}remaining_preys_ga.pkl', 'rb') as f:
        ga = pickle.load(f)
    with open(f'{save_path}remaining_preys_random.pkl', 'rb') as f:
        random = pickle.load(f)
    with open(f'{save_path}remaining_preys_ml.pkl', 'rb') as f:
        ml = pickle.load(f)
    all_methods = {}
    all_methods['GA'] = ga
    all_methods['Random'] = random
    all_methods['ML'] = ml
    with open('f{save_path}remaining_preys_all.pkl', 'wb') as f:
        pickle.dump(all_methods, f)


def deep_average_scores(scores_dict):
    """
    Recursively calculates the deep average of scores across all levels of a nested dictionary.
    """
    if isinstance(scores_dict, dict):
        # If it's a dictionary, recurse or aggregate scores
        scores = []
        for key in scores_dict:
            result = deep_average_scores(scores_dict[key])
            if isinstance(result, list):
                scores.extend(result)  # Extend if result is a list (this line might not be necessary)
            else:
                scores.append(result)  # Append the result directly if it's a single value
        return np.mean(scores)  # Calculate and return the mean of collected scores
    else:
        # If it's not a dictionary, it's assumed to be a single numeric value
        return scores_dict  # Return the numeric value directly

def process_file(file_path):
    """
    Processes a single pickle file, returning a dictionary of average scores for each method.
    Adjusts scores for files where lower values are better by negating the scores.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        average_scores = {}
        for method, scores in data.items():
            if method == 'ML':  # Handle 'ML' differently because it's one level deeper
                for sub_method, sub_scores in scores.items():
                    avg_score = deep_average_scores(sub_scores)
                    # Check if this is a file where lower scores are better
                    if 'nmf_scores_kl_all' in file_path or 'outlier_counts' in file_path:
                        avg_score = -avg_score  # Negate scores for these files
                    average_scores[sub_method] = avg_score
            else:
                avg_score = deep_average_scores(scores)
                # Check if this is a file where lower scores are better
                if 'nmf_scores_kl_all' in file_path or 'outlier_counts' in file_path:
                    avg_score = -avg_score  # Negate scores for these files
                average_scores[method] = avg_score
        return average_scores




def plot_values(normalized_df, save_path):
    colormap = plt.cm.Blues
    bar_height = 0.8 # Height of the horizontal bar

    plt.figure(figsize=(20, 12))
    ax = plt.gca()  # Get the current Axes instance

    # Background and separation aesthetics
    for i in range(0, len(normalized_df.index), 2):
        ax.axhspan(i-0.5, i+0.5, facecolor='grey', alpha=0.3)
    separation_regions = [(3, 4), (8, 9)]
    for start, end in separation_regions:
        ax.axvspan(start+0.4, end-0.4, facecolor='grey', alpha=0.3)

    # Find the index of the "Overall Score" column
    overall_score_index = normalized_df.columns.get_loc("Overall Score")

    # Plot normalized values
    for i, method in enumerate(normalized_df.index):
        text_color = 'black' if method == 'Random' else 'white' 
        for j, metric in enumerate(normalized_df.columns):
            value = normalized_df.loc[method, metric]
            if metric == 'Overall Score':
                # Correctly position the horizontal bar for the Overall Score
                ax.barh(y=i, width=value, left=overall_score_index-0.2, height=bar_height, color=colormap(value), edgecolor='black')
                # Annotate the bar with the actual value
                plt.text(overall_score_index, i, f"{value:.2f}", fontsize=14, ha='center', va='center', color=text_color)
            else:
                # Plot a circle for other metrics
                plt.scatter(x=j, y=i, s=1700, c=[colormap(value)], alpha=0.9, edgecolors='black', marker='o')
                # Annotate the plot with the actual value
                plt.text(j, i, f"{value:.2f}", fontsize=14, ha='center', va='center', color=text_color)

    # Final plot adjustments
    plt.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1)), ax=ax, label='Normalized Score', shrink=0.4)
    plt.yticks(ticks=np.arange(len(normalized_df.index)), labels=normalized_df.index, fontsize=18)
    plt.xticks(ticks=np.arange(len(normalized_df.columns)), labels=normalized_df.columns, rotation=90, fontsize=18)
    # plt.xlabel('Metrics', fontsize=18)
    # plt.ylabel('Methods', fontsize=18)
    plt.title('Dataset 1: Go et al., 2021', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{save_path}combined_metrics_comparison_plot_values_nmf_excluded.png', dpi=300)
    plt.savefig(f'{save_path}combined_metrics_comparison_plot_values_nmf_excluded.svg', dpi=300)
    plt.savefig(f'{save_path}combined_metrics_comparison_plot_values_nmf_excluded.pdf', dpi=300)
    plt.clf()

def create_combined_metrics_plot(save_path):
    combine_pkl_files(save_path)
    
    pickle_files = [f'{save_path}nmf_scores_all.pkl', f'{save_path}nmf_scores_ari_all.pkl', f'{save_path}nmf_scores_cos_all.pkl', f'{save_path}nmf_scores_kl_all.pkl', f'{save_path}nmf_scores_go_components_all.pkl', f'{save_path}remaining_preys_all.pkl', f'{save_path}go_results.pkl',f'{save_path}gmm_results.pkl', f'{save_path}gmm_hard_results.pkl', f'{save_path}leiden_results.pkl']
    combined_df = pd.DataFrame()

    # Process each file and update the DataFrame with the average scores
    for pickle_file in pickle_files:
        file_averages = process_file(pickle_file)
        for method, avg_score in file_averages.items():
            # Simplify column name by removing path and extension
            column_name = f'{pickle_file.split("/")[-1][:-4]}_Score'
            # Check if the file requires score inversion
            # if 'nmf_scores_kl_all' in pickle_file:
            #     avg_score = -avg_score  # Negate scores for this file
            combined_df.at[method, column_name] = avg_score

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

    metrics_name_mapping = {
        'remaining_preys_all_Score': 'Percenatage of remaining preys',
        'go_results_Score': 'GO terms retrieval percentage',
        'gmm_results_Score': 'GMM mean Pearson correlation',
        'gmm_hard_results_Score': 'GMM ARI',
        'leiden_results_Score': 'Leiden ARI',
        'nmf_scores_all_Score': 'NMF mean Pearson correlation',
        'nmf_scores_ari_all_Score': 'NMF ARI',
        'nmf_scores_cos_all_Score': 'NMF mean Cosine similarity',
        'nmf_scores_kl_all_Score': 'NMF mean KL divergence',
        'nmf_scores_go_components_all_Score': 'NMF mean GO Jaccard index',
        'Overall Score': 'Overall Score'
            
        }
    combined_df = combined_df.rename(index=methods_name_mapping)
    combined_df = combined_df.rename(columns=metrics_name_mapping)

    # Normalize the DataFrame (excluding the Overall Score and NMF mean correlation score)
    exclude_columns = ['NMF mean Pearson correlation']  # Add other columns to exclude if necessary
    columns_for_normalization = [col for col in combined_df.columns if col not in exclude_columns]
    normalized_df = combined_df[columns_for_normalization].fillna(0).apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    # Calculate and normalize the overall score for each method, excluding the NMF mean correlation score
    normalized_df['Overall Score'] = normalized_df.mean(axis=1)
    normalized_df['Overall Score'] = (normalized_df['Overall Score'] - normalized_df['Overall Score'].min()) / (normalized_df['Overall Score'].max() - normalized_df['Overall Score'].min())
    normalized_df.to_csv('plots/all_metrics.csv')

    # # Normalize the DataFrame (excluding the Overall Score)
    # normalized_df = combined_df.fillna(0).apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    # # Calculate and normalize the overall score for each method
    # normalized_df['Overall Score'] = normalized_df.mean(axis=1)
    # normalized_df['Overall Score'] = (normalized_df['Overall Score'] - normalized_df['Overall Score'].min()) / (normalized_df['Overall Score'].max() - normalized_df['Overall Score'].min())
    # normalized_df.to_csv('plots/all_metrics.csv')

     # Custom sorting to ensure 'GA' and 'Random' are always at the top
    ga_df = normalized_df.loc[['GA']]
    random_df = normalized_df.loc[['Random']]
    rest_df = normalized_df.drop(['GA', 'Random'])
    rest_df_sorted = rest_df.sort_values(by='Overall Score', ascending=False)
    sorted_normalized_df = pd.concat([rest_df_sorted, random_df, ga_df])

    # plot_with_ranks(sorted_normalized_df)
    plot_values(sorted_normalized_df, save_path)




