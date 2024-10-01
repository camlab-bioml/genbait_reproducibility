import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_storage import load_genetic_algorithm_results
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def plot_max_values_for_baits_boxplot_seeds(plots_path, ga_seeds_path):
    start_bait = 30
    end_bait = 80
    num_seeds = 10
    bait_numbers = list(range(start_bait, end_bait + 1))
    pickle_file_path = f'{plots_path}/boxplot_GA.pkl'

    # Check if the pickle file already exists
    if os.path.exists(pickle_file_path):
        # Load the DataFrame from the pickle file
        df_max_values = pd.read_pickle(pickle_file_path)
    else:
        # Initialize an empty DataFrame with seed numbers as indices and bait numbers as columns
        df_max_values = pd.DataFrame(index=range(num_seeds), columns=bait_numbers)

        for bait in bait_numbers:
            for seed in range(num_seeds):
                # Construct file paths for the current bait number and seed
                pop_path = f'{ga_seeds_path}popfile_features_{bait}_seed_{seed}.pkl'
                logbook_path = f'{ga_seeds_path}logbookfile_features_{bait}_seed_{seed}.pkl'
                hof_path = f'{ga_seeds_path}GA_number_of_baits_seeds/hoffile_features_{bait}_seed_{seed}.pkl'

                # Load genetic algorithm results
                pop, logbook, hof = load_genetic_algorithm_results(pop_path, logbook_path, hof_path)

                # Extract the max values from the logbook and filter out -inf values
                _, _, _, max_ = logbook.select("gen", "avg", "min", "max")
                filtered_max = [val for val in max_ if val != float('-inf')]
                
                # Since we're plotting boxplots, we take the last (maximum) value for simplicity
                if filtered_max:  # Check if filtered_max is not empty
                    df_max_values.at[seed, bait] = max(filtered_max)

        # Save the DataFrame as a pickle file
        df_max_values.to_pickle(pickle_file_path)

    ga_patch = mpatches.Patch(color='blue', label='GA')
    random_patch = mpatches.Patch(color='Orange', label='Random')


    # For plotting, convert the DataFrame back to a list of lists, handling missing data if necessary
    all_max_values = [df_max_values[col].dropna().tolist() for col in df_max_values]

    flierprops = dict(marker='', markersize=0)  # This hides the outliers

    plt.figure(figsize=(15, 6))
    plt.boxplot(all_max_values, positions=bait_numbers, flierprops=flierprops,
            patch_artist=True,  # This fills the boxes with color
            boxprops=dict(facecolor='lightblue')  # facecolor determines the fill
            ) 
    
    plt.ylim(0, 1)  # Adjust this based on your data range
    plt.xlabel('Number of Baits')
    plt.ylabel('Mean NMF Scores')
    plt.title('Boxplot of NMF Scores for Different Number of Baits Across Seeds in GA')
    plt.xticks(bait_numbers)
    # plt.grid(True)
    plt.savefig(f'{plots_path}nbaits vs. max value seeds boxplot GA.png', dpi=300)
    plt.savefig(f'{plots_path}nbaits vs. max value seeds boxplot GA.svg', dpi=300)
    plt.clf()

    # Load the DataFrame from the pickle file
    fitness_df = pd.read_pickle(f'{plots_path}boxplot_random.pkl')  # Replace with the correct path to your pickle file

    # Plotting the boxplot
    plt.figure(figsize=(15, 6))
    # Assuming that the DataFrame's columns are bait numbers and rows are seed values
    bait_numbers = fitness_df.columns.tolist()
    fitness_values = [fitness_df[bait].dropna() for bait in bait_numbers]

    plt.boxplot(fitness_values, positions=bait_numbers, flierprops=flierprops,
            patch_artist=True,  # This fills the boxes with color
            boxprops=dict(facecolor='lightblue')) 
    
    plt.ylim(0, 1)  # Adjust this based on your data range
    plt.xlabel('Number of Baits')
    plt.ylabel('Mean NMF Scores')
    plt.title('Boxplot of NMF Scores for Different Number of Baits Across Seeds in Random')
    plt.xticks(bait_numbers)
    # plt.grid(True)
    plt.savefig(f'{plots_path}nbaits vs. max value seeds boxplot Random.png', dpi=300)
    plt.savefig(f'{plots_path}nbaits vs. max value seeds boxplot Random.svg', dpi=300)
    plt.clf()


    # # Convert the DataFrames back to lists of lists, handling missing data if necessary
    # ga_values = [df_max_values[col].dropna().tolist() for col in df_max_values.columns]
    # random_values = [fitness_df[col].dropna().tolist() for col in fitness_df.columns]

    # Set up the figure
    plt.figure(figsize=(15, 6))

    # Set positions for each group to be side by side
    positions_ga = np.array(range(len(all_max_values))) * 2.0 
    positions_random = np.array(range(len(fitness_values))) * 2.0 

    # Plotting
    plt.boxplot(all_max_values, positions=positions_ga, patch_artist=True, 
                boxprops=dict(facecolor='blue', color='blue'),
                whiskerprops=dict(color='black'), capprops=dict(color='black'),
                flierprops=dict(marker='o', color='black', markersize=2),
                medianprops=dict(color='black'), widths=1)

    plt.boxplot(fitness_values, positions=positions_random, patch_artist=True, 
                boxprops=dict(facecolor='orange', color='orange'),
                whiskerprops=dict(color='black'), capprops=dict(color='black'),
                flierprops=dict(marker='o', color='black', markersize=2),
                medianprops=dict(color='black'), widths=1)


    # Customizations
    plt.xlabel('Number of baits', fontsize=16)
    plt.ylabel("NMF mean Pearson correlation score", fontsize = 16)
    # plt.title('Dataset 1: Go et al., 2021', fontsize=16)
    plt.xticks(np.arange(min(positions_ga), max(positions_ga)+1, 2.0), df_max_values.columns)
    plt.legend(handles=[ga_patch, random_patch], loc='lower right')
    # plt.grid(True)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(f'{plots_path}boxplot_GA_vs_Random.png', dpi=300)
    plt.savefig(f'{plots_path}boxplot_GA_vs_Random.svg', dpi=300)
    plt.savefig(f'{plots_path}boxplot_GA_vs_Random.pdf', dpi=300)
    plt.clf()

# def heatmap_number_of_baits():
#     methods = ['GA', 'chi_2', 'f_classif', 'mutual_info_classif', 'lasso', 'ridge', 'elastic_net', 'rf', 'gbm', 'xgb', 'random']
#     baits = list(range(30, 81))  # Assuming baits from 30 to 80 inclusive
#     averages = pd.DataFrame(index=methods, columns=baits)

#     for method in methods:
#         file_path = f'plots/boxplot_{method}.pkl'
#         df = pd.read_pickle(file_path)
#         averages.loc[method] = df.mean()

#     # Convert the averages to numeric values
#     averages = averages.apply(pd.to_numeric)

#     # Normalize the DataFrame for color mapping
#     normalized_df = (averages - averages.min().min()) / (averages.max().max() - averages.min().min())

#     colormap = plt.cm.Blues  # Choose a colormap
#     dot_size = 200  # Adjust dot size as needed

#     plt.figure(figsize=(25, 10))
#     ax = plt.gca()

#     # Plotting each cell with a dot
#     for i, method in enumerate(averages.index):
#         for j, bait in enumerate(averages.columns):
#             value = normalized_df.loc[method, bait]
#             plt.scatter(x=j, y=i, s=dot_size, c=[colormap(value)], edgecolors='black', marker='o')
#             plt.text(j, i, f"{averages.loc[method, bait]:.2f}", fontsize=8, ha='center', va='center')

#     # Customizing the axes
#     plt.xticks(ticks=np.arange(len(averages.columns)), labels=averages.columns, rotation=90)
#     plt.yticks(ticks=np.arange(len(averages.index)), labels=averages.index)
#     plt.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=1)), ax=ax, label='Normalized Average Value')
#     plt.xlabel('Number of Baits')
#     plt.ylabel('Methods')
#     plt.yticks(rotation=0)  # Keep the method names horizontal for better readability
#     plt.savefig('plots/number_of_baits_heatmap_nmf_scores.png')

