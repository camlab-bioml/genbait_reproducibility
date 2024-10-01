import matplotlib.pyplot as plt
from data_storage import load_genetic_algorithm_results
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def plot_max_values_for_baits():
    """
    Plots the maximum values from genetic algorithm results for a range of bait numbers.

    :param start_bait: The starting number of baits.
    :param end_bait: The ending number of baits.
    """
    start_bait = 30
    end_bait = 80
    max_values = []
    bait_numbers = list(range(start_bait, end_bait + 1))

    for bait in bait_numbers:
        # Construct file paths for the current bait number
        pop_path = f'GA_number_of_baits/popfile{bait}.pkl'
        logbook_path = f'GA_number_of_baits/logbookfile{bait}.pkl'
        hof_path = f'GA_number_of_baits/hoffile{bait}.pkl'

        # Load genetic algorithm results
        pop, logbook, hof = load_genetic_algorithm_results(pop_path, logbook_path, hof_path)

        # Extract the max value from the logbook
        _, _, _, max_ = logbook.select("gen", "avg", "min", "max")
        max_values.append(max(max_))  # Append the highest max value

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(bait_numbers, max_values, c='blue')
    plt.xlabel('Number of Baits')
    plt.ylabel('Max Value')
    plt.title('Max Values for Different Number of Baits')
    plt.xticks(bait_numbers)  # Ensure x-axis ticks match the bait numbers
    plt.savefig('plots/nbaits vs max value.png')
    plt.clf()

def plot_max_values_for_baits_boxplot():
    """
    Plots boxplots of the maximum values from genetic algorithm results for each bait number.
    """
    start_bait = 30
    end_bait = 80
    all_max_values = []
    bait_numbers = list(range(start_bait, end_bait + 1))

    for bait in bait_numbers:
        # Construct file paths for the current bait number
        pop_path = f'GA_number_of_baits/popfile{bait}.pkl'
        logbook_path = f'GA_number_of_baits/logbookfile{bait}.pkl'
        hof_path = f'GA_number_of_baits/hoffile{bait}.pkl'

        # Load genetic algorithm results
        pop, logbook, hof = load_genetic_algorithm_results(pop_path, logbook_path, hof_path)

        # Extract the max values from the logbook and filter out -inf values
        _, _, _, max_ = logbook.select("gen", "avg", "min", "max")
        filtered_max = [val for val in max_ if val != float('-inf')]
        all_max_values.append(filtered_max)

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.boxplot(all_max_values, positions=bait_numbers, showfliers=False, 
                boxprops=dict(color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'))
    plt.ylim(0.9, 1)
    plt.xlabel('Number of Baits')
    plt.ylabel('Max Values')
    plt.title('Boxplot of Max Values for Different Number of Baits')
    plt.xticks(bait_numbers)  # Ensure x-axis ticks match the bait numbers
    plt.grid(True)
    plt.savefig('plots/nbaits vs max value boxplot.png')
