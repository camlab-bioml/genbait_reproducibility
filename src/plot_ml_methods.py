import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import pickle
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def mean_min_component_correlation_plot_all_methods(original_dataframe, baits_df, number_of_components, subset_range, seed_number, save_path='plots', results_path='ML_results', top_features_path='top_features_ML'):
    original_data = original_dataframe.to_numpy()
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("hls", len(baits_df.columns))
    
    all_means = []
    bait_selections = {}
    
    all_mean_values = []  # Will store mean values for all columns
    all_min_values = []  # Will store min values for all columns

    
    for color, column in zip(colors, baits_df.columns): # type: ignore
        baits = baits_df[column].values
        mean_values = []
        min_values = []
        nmf_original = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
        scores_matrix_original = nmf_original.fit_transform(original_data)
        basis_matrix_original = nmf_original.components_.T
        
        for i in range(subset_range[0], subset_range[1] + 1):
            subset_indices = list(original_dataframe.index.get_indexer(baits[:i]))
            subset_data = original_data[subset_indices, :]
            
            nmf_subset = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
            nmf_subset.fit(subset_data)
            basis_matrix_subset = nmf_subset.components_.T

            cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
            cost_matrix = 1 - cosine_similarity
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

            corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)
            diagonal = corr_matrix[:number_of_components, number_of_components:]
            diagonal_mean = np.mean(np.diag(diagonal))
            mean_values.append(diagonal_mean)
            diagonal_min = np.min(np.diag(diagonal))
            min_values.append(diagonal_min)

        all_mean_values.append(mean_values)
        all_min_values.append(min_values)

        all_means.extend(mean_values)
        bait_selections[column] = {'baits': baits, 'mean_values': mean_values, 'min_values': min_values}
        
        plt.plot(range(subset_range[0], subset_range[1] + 1), mean_values, label=f'{column} mean', color=color)
        plt.plot(range(subset_range[0], subset_range[1] + 1), min_values, label=f'{column} min', color=color, linestyle='dashed')

    # Filtering and choosing the best baits
    top_5_percent_threshold = np.percentile(all_means, 90)
    bait_rankings = []

    for column, data in bait_selections.items():
        for i, (mean_val, min_val) in enumerate(zip(data['mean_values'], data['min_values'])):
            if min_val > 0 and mean_val >= top_5_percent_threshold:
                bait_rankings.append((mean_val, min_val, data['baits'][:i + subset_range[0]]))

    # Sort baits based on minimum correlation (primary) and mean correlation (secondary)
    sorted_baits = sorted(bait_rankings, key=lambda x: (x[1], x[0]), reverse=True)

    top_10_baits = sorted_baits[:10] if len(sorted_baits) > 10 else sorted_baits
    if not top_10_baits:
        print("Warning: No best baits determined. Returning the first set of baits as default.")
        first_column = baits_df.columns[0]
        top_10_baits = [(None, None, bait_selections[first_column]['baits'][:subset_range[0]])]

    # Save the top 10 best baits across all methods for 10 seeds as CSV
    for idx, (_, _, baits) in enumerate(top_10_baits, 1):
        pd.DataFrame(baits, columns=['selected_baits']).to_csv(f"{top_features_path}/top {idx} ML_baits_seed{seed_number}.csv", index=False)

    plt.xlabel('Number of Baits')
    plt.ylabel('Mean and minimum value of corresponding components correlation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.title('Mean and Minimum Component Correlation')
    plt.tight_layout()
    plt.savefig(f"{save_path}")
    # plt.show()
    plt.clf()


    # Return the top bait from the sorted list
    return top_10_baits[0][2], bait_selections


    


def aggregate_mean_min_across_seeds(all_seed_results, subset_range, methods, save_path='plots'):
    # Initialize dictionaries to store total mean and min values for each bait number and method
    total_means = {method: [] for method in methods}
    total_mins = {method: [] for method in methods}

    # Accumulate mean and min values for each method and bait number across all seeds
    for seed_result in all_seed_results:
        for method in methods:
            total_means[method].append(seed_result[method]['mean_values'])
            total_mins[method].append(seed_result[method]['min_values'])

    # Convert lists to numpy arrays for easier calculations
    for method in methods:
        total_means[method] = np.array(total_means[method]) # type: ignore
        total_mins[method] = np.array(total_mins[method]) # type: ignore

    # Calculate the average and standard deviation for each bait number and method
    avg_means = {method: np.mean(total_means[method], axis=0) for method in methods}
    avg_mins = {method: np.mean(total_mins[method], axis=0) for method in methods}
    std_dev_means = {method: np.std(total_means[method], axis=0) for method in methods}
    std_dev_mins = {method: np.std(total_mins[method], axis=0) for method in methods}

    # Plot the results
    plt.figure(figsize=(10, 8))
    for method, color in zip(methods, sns.color_palette("hls", len(methods))):  # Use seaborn to get distinct colors # type: ignore
        bait_numbers = range(subset_range[0], subset_range[1] + 1)
        plt.plot(bait_numbers, avg_means[method], label=f'{method} mean', color=color)
        plt.fill_between(bait_numbers, np.clip(avg_means[method] - std_dev_means[method], -1, 1), np.clip(avg_means[method] + std_dev_means[method], -1, 1), color=color, alpha=0.2)
        plt.plot(bait_numbers, avg_mins[method], label=f'{method} min', linestyle='dashed', color=color)
        plt.fill_between(bait_numbers, np.clip(avg_mins[method] - std_dev_mins[method], -1, 1), np.clip(avg_mins[method] + std_dev_mins[method], -1, 1), color=color, alpha=0.2)

    # Ensure the y-axis does not exceed the range [0, 1]
    # plt.ylim(-0.2, 1)
    # Add labels, legend, grid, title and save the figure
    plt.xlabel('Number of Baits')
    plt.ylabel('Average Mean and Minimum Value of Component Correlation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.title('Average Mean and Minimum Component Correlation Across Seeds with Uncertainty')
    plt.tight_layout()
    plt.savefig(f"{save_path}/ml_correlation_plot_averaged.png")
    plt.close()  # Close the plot to avoid displaying it inline if not desired
    


def plot_boxplots_baits_ml(all_seed_results, subset_range, methods, save_path='plots'):
    for method in methods:
        pickle_file_path = f"{save_path}/boxplot_{method}.pkl"

        # Check if the pickle file already exists
        if os.path.exists(pickle_file_path):
            # Load the DataFrame from the pickle file
            with open(pickle_file_path, 'rb') as file:
                data = pickle.load(file)
        else:
            # If the pickle file does not exist, calculate the data
            total_means = {i: [] for i in range(subset_range[0], subset_range[1] + 1)}

            # Accumulate mean values for the method and bait number across all seeds
            for seed_result in all_seed_results:
                for i in range(subset_range[0], subset_range[1] + 1):
                    mean_values = seed_result[method]['mean_values']
                    if i - subset_range[0] < len(mean_values):
                        total_means[i].append(mean_values[i - subset_range[0]])

            # Prepare data for boxplot
            data = pd.DataFrame({f'Bait {i}': total_means[i] for i in range(subset_range[0], subset_range[1] + 1)})

            # Save the DataFrame as a pickle file
            with open(pickle_file_path, 'wb') as file:
                pickle.dump(data, file)

        # Generate the boxplot
        # plt.figure(figsize=(12, 8))
        # sns.boxplot(data=data)
        # plt.xlabel('Number of Baits')
        # plt.ylabel('Average Diagonal NMF Score')
        # plt.title(f'Boxplot of Average Diagonal NMF Scores for {method}')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig(f"{save_path}/boxplot_{method}.png")
        # plt.clf()
        bait_numbers = list(range(30, 81))
        flierprops = dict(marker='o', color='lightgray', markersize=0.5) 
        plt.figure(figsize=(15, 6))
        plt.boxplot(data, positions=bait_numbers, flierprops=flierprops, 
                    boxprops=dict(color='blue'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'))
        plt.ylim(-0.2, 1)  # Adjust this based on your data range
        plt.xlabel('Number of Baits')
        plt.ylabel('Max NMF Scores')
        plt.title(f'Boxplot of Max Values for Different Number of Baits Across Seeds in {method}')
        plt.xticks(bait_numbers)  # Ensure x-axis ticks match the bait numbers
        plt.grid(True)
        plt.savefig(f'plots/nbaits vs. max value seeds boxplot {method}.png')


def plot_boxplots_baits_ml_pkl(plots_path='plots'):
    methods = ['chi_2','f_classif','mutual_info_classif','lasso','ridge','elastic_net','rf','gbm','xgb']  # Assuming df_norm contains method columns

    for method in methods:
        pickle_file_path = f"{plots_path}boxplot_{method}.pkl"

        # Check if the pickle file already exists
        if os.path.exists(pickle_file_path):
            # Load the DataFrame from the pickle file
            with open(pickle_file_path, 'rb') as file:
                data = pickle.load(file)

        bait_numbers = list(range(30, 81))
        # flierprops = dict(marker='o', color='lightgray', markersize=0.5) 
        flierprops = dict(marker='', markersize=0)  # This hides the outliers

        plt.figure(figsize=(15, 6))
        
        plt.boxplot(data, positions=bait_numbers, flierprops=flierprops,
            patch_artist=True,  # This fills the boxes with color
            boxprops=dict(facecolor='lightblue')) 
        

        plt.ylim(0, 1)  # Adjust this based on your data range
        plt.xlabel('Number of Baits')
        plt.ylabel('Max NMF Scores')
        plt.title(f'Boxplot of Max Values for Different Number of Baits Across Seeds in {method}')
        plt.xticks(bait_numbers)  # Ensure x-axis ticks match the bait numbers
        plt.grid(True)
        plt.savefig(f'{plots_path}nbaits vs. max value seeds boxplot {method}.png')