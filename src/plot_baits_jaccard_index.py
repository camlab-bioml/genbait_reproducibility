import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def jaccard_index(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def read_ga_genes(length, seed, ga_directory):
    filename = f"{ga_directory}/top_1_features_{length}_seed_{seed}.csv"
    df = pd.read_csv(filename)
    return df.iloc[:, 0].tolist()

def read_ml_genes(method, length, seed, ml_directory):
    filename = f"{ml_directory}/selected_baits_with_training_train-test-0.8_seed{seed}.csv"
    df = pd.read_csv(filename)
    return df[method][:length].tolist()

def read_random_genes(length, seed, random_directory):
    filename = f"{random_directory}/all_random_baits.csv"
    df = pd.read_csv(filename)
    # Calculate the column index based on length and seed
    # (length - 30) * 10 gives the starting column for a given length,
    # and seed specifies the exact column for that length and seed.
    col_index = (length - 30) * 10 + seed
    return df.iloc[:, col_index].dropna().tolist()

def plot_baits_jaccard_index():
    methods = ['GA', 'Random', 'chi_2', 'f_classif', 'mutual_info_classif', 'lasso', 'ridge', 'elastic_net', 'rf', 'gbm', 'xgb']
    jaccard_matrix = np.zeros((len(methods), len(methods)))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods[i:], start=i):
            jaccard_values = []
            for length in range(30, 81):
                for seed in range(10):
                    if method1 == 'GA':
                        genes1 = read_ga_genes(length, seed)
                    elif method1 == 'Random':
                        genes1 = read_random_genes(length, seed)
                    else:
                        genes1 = read_ml_genes(method1, length, seed)

                    if method2 == 'GA':
                        genes2 = read_ga_genes(length, seed)
                    elif method2 == 'Random':
                        genes2 = read_random_genes(length, seed)
                    else:
                        genes2 = read_ml_genes(method2, length, seed)

                    jaccard_values.append(jaccard_index(genes1, genes2))
            jaccard_matrix[i, j] = jaccard_matrix[j, i] = np.mean(jaccard_values)

    plt.figure(figsize=(10,10))
    sns.heatmap(jaccard_matrix, annot=True, fmt=".3f", xticklabels=methods, yticklabels=methods, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Jaccard Index Heatmap Between Methods')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig('plots/biats_overlap_jaccrad_heatmap.png')

# Now you can simply call the function
# plot_baits_jaccard_index()


def plot_seed_comparison_heatmaps(ga_path, ml_path, random_path, save_path):
    methods = ['GA', 'Random', 'chi_2', 'f_classif', 'mutual_info_classif', 'lasso', 'ridge', 'elastic_net', 'rf', 'gbm', 'xgb']
    seed_range = range(10)
    length_range = range(30, 81)  # Assuming lengths from 30 to 80

    for method in methods:
        # Initialize a matrix to hold the average Jaccard indices for each seed pair
        jaccard_matrix = np.zeros((len(seed_range), len(seed_range)))

        for i, seed1 in enumerate(seed_range):
            for j, seed2 in enumerate(seed_range):
                if i <= j:  # To avoid recalculating for symmetric pairs
                    jaccard_values = []
                    for length in length_range:
                        if method == 'GA':
                            genes1 = read_ga_genes(length, seed1, ga_path)
                            genes2 = read_ga_genes(length, seed2, ga_path)
                        elif method == 'Random':
                            genes1 = read_random_genes(length, seed1, random_path)
                            genes2 = read_random_genes(length, seed2, random_path)
                        else:
                            genes1 = read_ml_genes(method, length, seed1, ml_path)
                            genes2 = read_ml_genes(method, length, seed2, ml_path)

                        jaccard_values.append(jaccard_index(genes1, genes2))

                    # Calculate the average Jaccard index for this pair of seeds across all lengths
                    avg_jaccard = np.mean(jaccard_values)
                    jaccard_matrix[i, j] = avg_jaccard
                    jaccard_matrix[j, i] = avg_jaccard  # Fill in the symmetric value

        # Plotting the heatmap for this method
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(jaccard_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=list(seed_range), yticklabels=list(seed_range), vmin=0, vmax=1, cbar=False)
        cbar = ax.figure.colorbar(ax.collections[0], ax=ax, location="right", shrink=0.2, aspect=10)
        cbar.set_label('Jaccard index', fontsize=12) 
        plt.title(f'Mean Jaccard index for {method}', fontsize=16)
        plt.xlabel('Seed', fontsize=16)
        plt.ylabel('Seed', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_path}jaccard_seed_comparison_{method}.png', dpi=300)
        plt.savefig(f'{save_path}jaccard_seed_comparison_{method}.svg', dpi=300)
        plt.savefig(f'{save_path}jaccard_seed_comparison_{method}.pdf', dpi=300)
        plt.close()  # Close to prevent inline display if in a notebook
