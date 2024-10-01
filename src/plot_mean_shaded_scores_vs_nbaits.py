import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def plot_nmf_mean_values_with_shades(plots_path):
    # Define the methods to be plotted
    methods = ['GA', 'random', 'chi_2', 'mutual_info_classif', 'f_classif', 'lasso', 'ridge', 'elastic_net', 'rf', 'gbm', 'xgb']

    # Map method names to more descriptive names
    methods_name_mapping = {
        'GA': 'GA',
        'random': 'Random',
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

    # Select colors from the 'tab20' colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(15, 6))

    for method, color in zip(methods, colors):
        # Load the DataFrame from pickle file
        df = pd.read_pickle(f'{plots_path}boxplot_{method}.pkl')
        
        # Calculate mean and standard deviation
        mean_values = df.mean()
        std_dev = df.std()

        # Define the x-axis range based on column names
        x = np.arange(30, 81)  # Assuming column names are integers from 30 to 80
        y = mean_values.values
        error = std_dev.values

        # Plot mean values and shaded standard deviation for each method using mapped names
        ax.plot(x, y, label=methods_name_mapping[method], color=color, linewidth=0.8)
        ax.fill_between(x, y - error, y + error, color=color, alpha=0.2)

    # # Set axes limits to remove white space
    # ax.set_xlim(x[0], x[-1])
    # ax.set_ylim(min(y - error), max(y + error))

    # Set axis labels and formatting
    ax.set_xlabel('Number of baits', fontsize=18)
    ax.set_ylabel("NMF mean Pearson correlation score", fontsize=18)
    # ax.set_title('Dataset 1: GO et al., 2021', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim(x[0] , x[-1])
    ax.set_ylim(0,1)
    # Set x-axis ticks with steps of 10
    ax.set_xticks(np.arange(30, 81, 10))
    ax.set_xticklabels(np.arange(30, 81, 10), fontsize=12) 
    
    plt.tight_layout()
    plt.savefig(f'{plots_path}nmf_mean_values_shaded_vs_nbaits.png', dpi=300)
    plt.savefig(f'{plots_path}nmf_mean_values_shaded_vs_nbaits.svg', dpi=300)
    plt.savefig(f'{plots_path}nmf_mean_values_shaded_vs_nbaits.pdf', dpi=300)

