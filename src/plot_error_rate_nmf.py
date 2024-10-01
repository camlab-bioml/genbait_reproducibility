import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def calculate_outliers(data):
    """Calculate outliers based on IQR."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound]
    return outliers

def calculate_error_rate(data):
    """Calculate the error rate (proportion of outliers)."""
    outliers = calculate_outliers(data)
    error_rate = len(outliers) / len(data) if data else 0
    return error_rate

def plot_error_rate_nmf(components_range, save_path='plots'):
    """Function to plot error rates of outliers for each method and number of components from actual data."""
    pickle_file = os.path.join(save_path, 'nmf_scores.pkl')

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            aggregated_diagonal_values = pickle.load(f)

        error_rates = {}
        for category, values in aggregated_diagonal_values.items():
            if category == 'Methods':
                for method, method_data in values.items():
                    error_rates[method] = [calculate_error_rate(method_data.get(n, [])) for n in components_range]
            else:
                error_rates[category] = [calculate_error_rate(values.get(n, [])) for n in components_range]

        plt.figure(figsize=(15, 6))
        bar_width = 0.15  # Increased bar width
        method_names = list(error_rates.keys())
        num_components = len(components_range)

        for i, comp_number in enumerate(components_range):
            # Adding extra space after each group
            positions = np.arange(len(method_names)) * (bar_width * num_components + 0.5) + i * bar_width
            comp_error_rates = [error_rates[method][i] for method in method_names]
            plt.bar(positions, comp_error_rates, width=bar_width, label=f'Comp. {comp_number}')

        plt.xlabel('Methods')
        plt.ylabel('Error Rate')
        plt.title('Error Rates of Outliers for Each Method by Number of Components')
        # Adjusting the x-ticks position
        plt.xticks(np.arange(len(method_names)) * (bar_width * num_components + 0.5) + bar_width * (num_components / 2 - 0.5), method_names)
        plt.legend()
        plt.savefig(f'{save_path}/nmf error rate - different components.png')
        plt.clf()

        plt.figure(figsize=(10, 6))  # Setting the figure size for the new plot
        sum_error_rates = {method: sum(error_rates[method]) for method in method_names}
        methods = list(sum_error_rates.keys())
        total_errors = list(sum_error_rates.values())

        plt.bar(methods, total_errors, color='skyblue')  # Creating a bar plot
        plt.xlabel('Methods')
        plt.ylabel('Sum of Error Rates')
        plt.title('Sum of Error Rates for Each Method Across All Components')

        plt.savefig(f'{save_path}/nmf error rate total.png')  # Saving the new plot
