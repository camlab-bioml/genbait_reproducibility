from matplotlib import patches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def load_and_process_gaf(file_path):
    # Load the GAF file into a DataFrame
    df = pd.read_csv(file_path, sep='\t', comment='!', header=None, dtype=str)
    
    # Set column names based on the GAF 2.1 specification
    column_names = [
        "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
        "DB_Reference", "Evidence_Code", "With_or_From", "Aspect",
        "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type",
        "Taxon", "Date", "Assigned_By", "Annotation_Extension",
        "Gene_Product_Form_ID"
    ]
    df.columns = column_names[:len(df.columns)]  # Handles cases where some optional columns might be missing
    
    # Calculate the term size for each GO term
    term_size = df.groupby('GO_ID')['DB_Object_Symbol'].nunique()
    term_size = term_size.reset_index()
    term_size.columns = ['GO_ID', 'Term_Size']
    
    # Get associated genes for each GO term
    associated_genes = df.groupby('GO_ID')['DB_Object_Symbol'].unique()
    associated_genes = associated_genes.reset_index()
    associated_genes.columns = ['GO_ID', 'Associated_Genes']
    
    # Merge the dataframes on 'GO_ID'
    merged_df = pd.merge(term_size, associated_genes, on='GO_ID')
    
    return df, merged_df


def get_go_cc_for_genes(df, genes, merged_df, datasets_path, max_term_size=10000):
    df_norm = pd.read_csv(f'{datasets_path}df_norm.csv', index_col=0)
    df_subset = df_norm.loc[genes]
    mask = (df_subset != 0).any(axis=0)
    df_subset_reduced = df_subset.loc[:, mask]
    subset_df = df[df['DB_Object_Symbol'].isin(df_subset_reduced.columns)]
    cc_df = subset_df[subset_df['Aspect'] == 'C']
    unique_go_cc_terms = cc_df['GO_ID'].unique()
    large_terms = merged_df[merged_df['Term_Size'] <= max_term_size]['GO_ID'].tolist()
    filtered_terms = [term for term in unique_go_cc_terms if term in large_terms]
    return filtered_terms





def process_ga_results(ga_path, datasets_path):
    df, merged_df = load_and_process_gaf(f'{datasets_path}goa_human.gaf')
    max_size = 10000
    primary_baits = pd.read_csv(f'{datasets_path}original_baits.csv', header=0).iloc[:,0].to_list()
    go_terms_original = get_go_cc_for_genes(df, primary_baits, merged_df, max_size)
    results_ga = {}
    directory = ga_path
    # Group files by number of features
    feature_groups = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            parts = file_name.split('_')
            num_features = int(parts[3])  # Extracting the number of features
            if num_features not in feature_groups:
                feature_groups[num_features] = []
            feature_groups[num_features].append(file_name)

    # Process each group of features
    for num_features, files in feature_groups.items():
        results_ga[num_features] = []
        for file_name in files:
            selected_baits_df = pd.read_csv(os.path.join(directory, file_name))
            selected_baits = selected_baits_df.iloc[:, 0].tolist()
            go_terms_subset = get_go_cc_for_genes(df, selected_baits, merged_df, datasets_path, max_size)
            overlap_subset_original = len(set(go_terms_subset) & set(go_terms_original)) / len(go_terms_original) * 100
            results_ga[num_features].append(overlap_subset_original)

    
    return results_ga


def process_random_baseline(random_path, datasets_path):
    df, merged_df = load_and_process_gaf(f'{datasets_path}goa_human.gaf')
    max_size = 10000
    primary_baits = pd.read_csv(f'{datasets_path}original_baits.csv', header=0).iloc[:,0].to_list()
    go_terms_original = get_go_cc_for_genes(df, primary_baits, merged_df, max_size)
    df_random_baits_all = pd.read_csv(f'{random_path}all_random_baits.csv', header=None)
    results_random = {length: [] for length in range(30, 81)}
    for bait_length in range(30, 81):
        column_start = (bait_length - 30) * 10
        column_end = column_start + 10

            
        for column in range(column_start, column_end):
            selected_baits = df_random_baits_all[column].dropna().tolist()
            go_terms_subset = get_go_cc_for_genes(df, selected_baits, merged_df, datasets_path, max_size)
            overlap_subset_original = len(set(go_terms_subset) & set(go_terms_original)) / len(go_terms_original) * 100
            results_random[bait_length].append(overlap_subset_original)
                 
    return results_random




def process_ml_results(ml_path, datasets_path):
    ml_names = ['chi_2', 'f_classif', 'mutual_info_classif', 'lasso', 'ridge', 'elastic_net', 'rf', 'gbm', 'xgb']
    df, merged_df = load_and_process_gaf(f'{datasets_path}goa_human.gaf')
    max_size = 10000
    primary_baits = pd.read_csv(f'{datasets_path}original_baits.csv', header=0).iloc[:,0].to_list()
    go_terms_original = get_go_cc_for_genes(df, primary_baits, merged_df, max_size)
    results_ml = {method: {entity_count: [] for entity_count in range(30, 81)} for method in ml_names}
    for file_name in os.listdir(ml_path):
        if file_name.endswith('.csv'):
            methods_df = pd.read_csv(os.path.join(ml_path, file_name))

            for method in ml_names:
                for entity_count in range(30, 81):
                    if method in methods_df:
                        selected_baits = methods_df[method].dropna().iloc[:entity_count].tolist()
                        go_terms_subset = get_go_cc_for_genes(df, selected_baits, merged_df, datasets_path, max_size)
                        overlap_subset_original = len(set(go_terms_subset) & set(go_terms_original)) / len(go_terms_original) * 100
                        results_ml[method][entity_count].append(overlap_subset_original)

    return results_ml


def aggregate_all_results(ga_path, ml_path, random_path, save_path, datasets_path):
    results_file=f'{save_path}go_results.pkl'
    # Process the results
    results_ga = process_ga_results(ga_path, datasets_path)
    results_random = process_random_baseline(random_path, datasets_path)
    results_ml = process_ml_results(ml_path, datasets_path)

    # Combine all results into a single dictionary
    combined_results = {
        'GA': results_ga,
        'Random': results_random,
        'ML': results_ml
    }


    # Serialize the combined results to a pickle file
    with open(results_file, 'wb') as f:
        pickle.dump(combined_results, f)

    print(f"Results saved to '{results_file}'.")



# Function to plot the methods' values without subplots
def plot_go(df_norm, ga_path, ml_path, random_path, save_path, datasets_path):
    results_file=f'{save_path}go_results.pkl'
    if not os.path.exists(results_file):
        aggregate_all_results(df_norm, ga_path, ml_path, random_path, save_path)

    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    flierprops = dict(marker='o', markersize=1, linestyle='none')

    # Prepare subplot structure
    n_methods = len(results['ML']) + 2  # +2 for GA and Random
    n_cols = 3
    n_rows = n_methods // n_cols + (n_methods % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4), constrained_layout=True)
    if n_rows == 1:
        axs = axs[np.newaxis, :]  # Ensure axs is 2D array for consistency
    elif n_cols == 1:
        axs = axs[:, np.newaxis]

    # Plot GA and Random first
    methods = ['GA', 'Random'] + list(results['ML'].keys())
    for idx, method in enumerate(methods):
        ax = axs[idx//n_cols, idx%n_cols]
        feature_nums = []
        values = []
        if method in ['GA', 'Random']:
            for feature_num, vals in results[method].items():
                feature_nums.extend([feature_num]*len(vals))
                values.extend(vals)
        else:
            for feature_num, vals in results['ML'][method].items():
                feature_nums.extend([feature_num]*len(vals))
                values.extend(vals)
        
        ax.boxplot([values[feature_nums.index(fn):feature_nums.index(fn)+feature_nums.count(fn)] for fn in sorted(set(feature_nums))], labels=sorted(set(feature_nums)))
        ax.set_title(method)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Values')

    # Hide any unused axes
    for ax in axs.flat[len(methods):]:
        ax.set_visible(False)

    plt.suptitle('Values Across Different Methods and Feature Numbers')
    plt.savefig(f'{save_path}GO terms retrieval percentage vs. number of baits.png', dpi=300)
    plt.savefig(f'{save_path}GO terms retrieval percentage vs. number of baits.svg', dpi=300)

    plt.clf()

    # Initial setup
    method_labels = ['GA', 'Random'] + list(results['ML'].keys())
    aggregated_data = []
    median_values = []

    # Aggregate data and calculate medians for GA and Random
    for method in ['GA', 'Random']:
        all_values = []
        for feature_data in results[method].values():
            all_values.extend(feature_data)
        aggregated_data.append(all_values)
        median_values.append(np.median(all_values))

    # Aggregate data and calculate medians for ML methods
    for ml_method in results['ML'].keys():
        all_values = []
        for feature_data in results['ML'][ml_method].values():
            all_values.extend(feature_data)
        aggregated_data.append(all_values)
        median_values.append(np.median(all_values))

    # Sort methods by median values
    sorted_indices = np.argsort(median_values)[::-1]  # Descending order
    sorted_aggregated_data = [aggregated_data[i] for i in sorted_indices]
    sorted_method_labels = [method_labels[i] for i in sorted_indices]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.boxplot(sorted_aggregated_data, labels=sorted_method_labels, patch_artist=True, flierprops=flierprops, boxprops=dict(facecolor='lightblue'))
    ax.set_xlabel('Methods')
    ax.set_ylabel('GO terms retrieval percentage')
    ax.set_title('Aggregated Values Across Methods Ordered by Median')
    plt.ylim(0, 100)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{save_path}GO terms retrieval percentage vs. each method (sorted).png', dpi=300)
    plt.savefig(f'{save_path}/GO terms retrieval percentage vs. each method (sorted).svg', dpi=300)

    plt.clf()


    # Initialize figure for plotting
    fig, ax = plt.subplots(figsize=(15, 9))

    # Define a color map for the methods
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))

    # Function to aggregate values by feature number
    def aggregate_values(method_data):
        aggregated_scores = {}
        for feature_num, vals in method_data.items():
            if feature_num not in aggregated_scores:
                aggregated_scores[feature_num] = []
            aggregated_scores[feature_num].extend(vals)
        return aggregated_scores

    # Plot data for each method
    for idx, method in enumerate(methods):
        if method in ['GA', 'Random']:
            method_data = results[method]
        else:
            method_data = results['ML'][method]

        # Aggregate values by feature number
        aggregated_scores = aggregate_values(method_data)

        # Sort feature numbers for plotting
        feature_numbers = sorted(aggregated_scores.keys())
        means = [np.mean(aggregated_scores[fn]) for fn in feature_numbers]
        std_devs = [np.std(aggregated_scores[fn]) for fn in feature_numbers]

        # Plot mean values with shaded areas for standard deviation
        ax.plot(feature_numbers, means, label=method, color=colors[idx])
        ax.fill_between(feature_numbers, [m - s for m, s in zip(means, std_devs)], [m + s for m, s in zip(means, std_devs)], color=colors[idx], alpha=0.2)

    # Set plot parameters
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Values')
    ax.set_title('Values Across Different Methods and Feature Numbers')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.05))
    ax.set_xlim(left=min(feature_numbers), right=max(feature_numbers))
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f'{save_path}GO terms retrieval percentage vs. number of baits_means shaded.png', dpi=300)
    plt.savefig(f'{save_path}GO terms retrieval percentage vs. number of baits_means shaded.svg', dpi=300)

    plt.clf()



