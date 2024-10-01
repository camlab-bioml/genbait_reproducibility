import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def identify_and_save_outliers(results, components_range, save_path):
    """
    Identify outliers in the NMF scores using the IQR method and save the counts.
    Handles the nested structure for 'GA', 'Random', and 'ML' methods.

    :param results: Dictionary of NMF scores with a specific nested structure
    :param save_path: Directory path to save the outlier count pickle files
    """
    outlier_counts = {'GA': {}, 'Random': {}, 'ML': {}}

    # Process 'GA' and 'Random' similarly
    for method in ['GA', 'Random']:
        for feature_count, comps in results[method].items():
            outlier_counts[method][feature_count] = {}
            for comp, values in comps.items():
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                # outliers = [x for x in values if x < lower_bound or x > upper_bound]
                outliers = [x for x in values if x < lower_bound]
                outlier_counts[method][feature_count][comp] = len(outliers)

    # Process 'ML' with its additional layer
    for ml_method, features_data in results['ML'].items():
        outlier_counts['ML'][ml_method] = {}
        for feature_count, comps in features_data.items():
            outlier_counts['ML'][ml_method][feature_count] = {}
            for comp, values in comps.items():
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                # outliers = [x for x in values if x < lower_bound or x > upper_bound]
                outliers = [x for x in values if x < lower_bound]
                outlier_counts['ML'][ml_method][feature_count][comp] = len(outliers)

    # Save the outlier counts to a pickle file
    full_path = os.path.join(save_path, 'outlier_counts.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(outlier_counts, f)

    print(f"Outlier counts saved to {full_path}")




    total_outliers = {'GA': 0, 'Random': 0, 'ML': 0}

    # For GA and Random
    for method in ['GA', 'Random']:
        for feature_list in results[method].values():
            for feature_dict in feature_list:
                total_outliers[method] += sum(feature_dict.values())

    # For ML methods, aggregate separately and then add to the 'ML' key
    ml_total = 0
    for ml_method, feature_data in results['ML'].items():
        for feature_list in feature_data.values():
            for feature_dict in feature_list:
                ml_total += sum(feature_dict.values())
    total_outliers['ML'] = ml_total


    # Prepare data for plotting
    methods = list(total_outliers.keys())
    outlier_counts = list(total_outliers.values())

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(methods, outlier_counts, color='skyblue')
    plt.xlabel('Methods')
    plt.ylabel('Total Number of Outliers')
    plt.title('Total Outliers by Method')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()


with open('plots/nmf_scores_all.pkl', 'rb') as f:
    nmf_results = pickle.load(f)

identify_and_save_outliers(nmf_results, (20,21), 'plots')
