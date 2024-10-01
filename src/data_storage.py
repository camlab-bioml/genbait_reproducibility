import pickle

def save_genetic_algorithm_results(population, logbook, hof, pop_file_path='popfile.pkl', logbook_file_path='logbookfile.pkl', hof_file_path='hoffile.pkl'):
    """
    Save the results of the genetic algorithm (population and logbook) to specified file paths using pickle.

    Args:
    - population (list): The population resulting from the GA.
    - logbook (deap.tools.Logbook): The logbook resulting from the GA.
    - pop_file_path (str): Path to save the population file.
    - logbook_file_path (str): Path to save the logbook file.
    - hof_file_path (str): Path to save the hof file.

    Returns:
    None
    """
    with open(pop_file_path, 'wb') as pop_file:
        pickle.dump(population, pop_file)
        
    with open(logbook_file_path, 'wb') as logbook_file:
        pickle.dump(logbook, logbook_file)

    with open(hof_file_path, 'wb') as hof_file:
        pickle.dump(hof, hof_file)

def load_genetic_algorithm_results(pop_file_path='popfile.pkl', logbook_file_path='logbookfile.pkl', hof_file_path='hoffile.pkl'):
    """
    Load previously saved population and logbook from their respective file paths.

    Args:
    - pop_file_path (str): Path to load the population file.
    - logbook_file_path (str): Path to load the logbook file.

    Returns:
    - population (list): The loaded population.
    - logbook (deap.tools.Logbook): The loaded logbook.
    """
    with open(pop_file_path, 'rb') as pop_file:
        population = pickle.load(pop_file)
        
    with open(logbook_file_path, 'rb') as logbook_file:
        logbook = pickle.load(logbook_file)

    with open(hof_file_path, 'rb') as hof_file:
        hof = pickle.load(hof_file)
        
    return population, logbook, hof
