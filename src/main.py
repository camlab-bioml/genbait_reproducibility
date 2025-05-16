import argparse
import pandas as pd
import os
from data_loading import load_data, preprocess_data
from genetic_algorithm import run_genetic_algorithm
from data_storage import save_genetic_algorithm_results, load_genetic_algorithm_results
from random_baseline import generate_random_baseline, generate_random_baseline_benchmark, generate_best_set_sequence_from_random, save_best_baits
from baits_with_highest_number_of_preys import baits_with_most_preys
from plot_GA_vs_random import plot_ga_vs_random
from save_top_features import get_and_save_top_features_from_ga, get_and_save_top_features_from_ga_seeds
from plot_components_correlation import plot_components_correlations_reordered
from plot_tsne import plot_tsne
from gsea_analysis import perform_gsea_analysis
from ml_feature_selection_methods import run_feature_selection_for_seeds
from plot_ml_methods import mean_min_component_correlation_plot_all_methods
from plot_ml_methods import aggregate_mean_min_across_seeds
from plot_ml_methods import plot_boxplots_baits_ml, plot_boxplots_baits_ml_pkl
from plot_nbaits_vs_maxvalue_seeds import plot_max_values_for_baits_boxplot_seeds
from genetic_algorithm_number_of_baits_seeds import run_ga_for_number_of_baits_and_seeds
from plot_preys_captured_by_baits import plot_preys_for_baits
from plot_remaining_preys import plot_remaining_preys
from plot_go import plot_go
from plot_knn_leiden_ari import plot_leiden_ari
from plot_gmm_ari import plot_gmm_hard_ari
from plot_gmm_correlation import plot_gmm_correlation
from plot_nmf_scores import plot_nmf_scores
from plot_nmf_ari_scores import plot_nmf_ari_scores
from plot_nmf_cosine_scores import plot_nmf_cos_scores
from plot_nmf_kl_scores import plot_nmf_kl_scores
from plot_nmf_go_component import plot_nmf_go_scores
from plot_mean_shaded_scores_vs_nbaits import plot_nmf_mean_values_with_shades
from deap import base, creator, tools
from plot_nmf_scores_min import plot_nmf_scores_min
from plot_nmf_cosine_scores_min import plot_nmf_cos_scores_min
from plot_nmf_kl_scores_min import plot_nmf_kl_scores_min
from plot_nmf_go_component_min import plot_nmf_go_scores_min
from plot_nmf_ari_scores_min import plot_nmf_ari_scores_min
from plot_combined_metrics import create_combined_metrics_plot
from plot_runtime_analysis import plot_runtime_analysis
from plot_individual_components_bait_size import plot_individual_components_vs_bait_sizes
from plot_topology_metrics import plot_topology_metrics
from cell_line_data_simulator_baits import plot_baits_expression_heatmap
from plot_nmf_score_vs_expression import plot_nmf_score_vs_expression
from cell_line_data_simulator_preys import generate_simulated_expression_data
from get_proteomicsdb_cell_line_data import get_cellline_data
from plot_nmf_scores_cmbined_datasets import plot_nmf_correlation_combined
from plot_nmf_min_scores_cmbined_datasets import plot_nmf_min_correlation_combined
from plot_nmf_cosine_scores_combined_datasets import plot_nmf_cos_combined
from plot_nmf_cosine_min_scores_combined_datasets import plot_nmf_min_cos_combined
from plot_nmf_kl_scores_combined_datasets import plot_nmf_kl_combined
from plot_nmf_kl_min_scores_combined_datasets import plot_nmf_max_kl_combined
from plot_nmf_ari_scores_combined_datasets import plot_nmf_ari_combined
from plot_nmf_ari_scores_min_combined_datasets import plot_nmf_min_purity_score_combined
from plot_nmf_go_components_combined_datasets import plot_nmf_go_combined
from plot_nmf_go_components_min_combined_datasets import plot_nmf_min_go_combined
from plot_remaining_preys_combined_datasets import plot_combined_remaining_preys
from plot_go_combined_datasets import plot_combined_go
from plot_knn_leiden_ari_combined_datasets import plot_leiden_combined
from plot_gmm_ari_combined_datasets import plot_gmm_ari_combined
from plot_gmm_correlation_combined_datasets import plot_gmm_combined
import yaml
import warnings
warnings.filterwarnings('ignore')



def main():
    
    # ✅ Define the --config argument properly
    parser = argparse.ArgumentParser(description='Bait Selection for BioID Map')
    parser.add_argument('--step', required=True, choices=[
        'load_data', 'run_ga', 'ga_evaluation', 'ga_number_of_baits_seeds',
        'seeds_evaluation', 'run_ml_methods', 'plot_ml_methods', 'plot_nmf_scores',
        'plot_nmf_cos_scores', 'plot_nmf_kl_scores', 'plot_nmf_ari_scores',
        'plot_nmf_go_scores', 'plot_nmf_scores_min', 'plot_nmf_cos_scores_min',
        'plot_nmf_kl_scores_min', 'plot_nmf_ari_scores_min', 'plot_nmf_go_scores_min',
        'remaining_preys_evaluation', 'leiden_evaluation', 'gmm_evaluation',
        'gmm_hard_evaluation', 'go_evaluation', 'runtime_analysis',
        'individual_components_correlation', 'topology_analysis',
        'bait_expression_analysis', 'simulation_expression_analysis',
        'combined_metrics',
        'combined_nmf_corr',
        'combined_nmf_min_corr',
        'combined_nmf_cos',
        'combined_nmf_min_cos',
        'combined_nmf_kl',
        'combined_nmf_min_kl',
        'combined_nmf_ari',
        'combined_nmf_min_purity',
        'combined_nmf_go',
        'combined_nmf_min_go'
        'combined_remaining_preys',
        'combined_go_retrieval',
        'combined_leiden',
        'combined_gmm_hard',
        'combined_gmm_correlation'
        'final_step'
    ], help='Which step to execute')

    # ✅ Add --config argument (fixing the error)
    parser.add_argument('--config', required=True, help='Path to the configuration file')

    args = parser.parse_args()

    # ✅ Load YAML config dynamically
    def load_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    CONFIG = load_config(args.config)


# Load Data
    if args.step == 'load_data':
        df = load_data(CONFIG['datasets_path']+'saint-latest.txt', sep='\t')
        # primary_baits_df = load_data('datasets/genesymbols_uniprotids_less_than_110.csv', index_col=0)
        # primary_baits = list(primary_baits_df['Gene Symbols'])
        # Preprocess Data
        df_norm = preprocess_data(df, file_path=CONFIG['datasets_path'])
        # Save df_norm
        df_norm.to_csv(CONFIG["df_norm_path"])

    # Run Genetic Algorithm and Save Results
    elif args.step == 'run_ga':
        # Load df_norm
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        
        pop, logbook, hof = run_genetic_algorithm(df_norm,
                                                  n_components=CONFIG["number_of_components"],
                                                  subset_range=CONFIG["subset_range_GA"],
                                                  population_size=CONFIG["population_size"],
                                                  n_generations=CONFIG["number_of_generations"],
                                                  cxpb=CONFIG["cxbp"],
                                                  mutpb=CONFIG["mutpb"])

        save_genetic_algorithm_results(pop, logbook, hof, 
                                       pop_file_path=f'{CONFIG["ga_results_path"]}/popfile.pkl', 
                                       logbook_file_path=f'{CONFIG["ga_results_path"]}/logbookfile.pkl', 
                                       hof_file_path=f'{CONFIG["ga_results_path"]}/hoffile.pkl')
    elif args.step == 'load_ga':
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # single-objective maximization problem
        creator.create("Individual", list, fitness=creator.FitnessMax) # type: ignore

        pop, logbook, hof = load_genetic_algorithm_results(f'{CONFIG["ga_results_path"]}/popfile.pkl',
                                                           f'{CONFIG["ga_results_path"]}/logbookfile.pkl',
                                                           f'{CONFIG["ga_results_path"]}/hoffile.pkl')  
    
    elif args.step == 'ga_evaluation':
        # Load df_norm
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # single-objective maximization problem
        creator.create("Individual", list, fitness=creator.FitnessMax) # type: ignore
        pop, logbook, hof = load_genetic_algorithm_results(f'{CONFIG["ga_results_path"]}/popfile.pkl',
                                                           f'{CONFIG["ga_results_path"]}/logbookfile.pkl',
                                                           f'{CONFIG["ga_results_path"]}/hoffile.pkl')  
        gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
        ga_selected_baits_list = get_and_save_top_features_from_ga(hof, df_norm, CONFIG['population_size'], CONFIG['number_of_generations'], save_path=CONFIG['top_features_GA_path'])
        ga_selected_baits = ga_selected_baits_list[0]
        random_fitnesses, best_random_selected_baits = generate_random_baseline(df_norm, CONFIG['number_of_components'], CONFIG['subset_range_random'], file_path=CONFIG['random_baseline'])
        best_set = generate_best_set_sequence_from_random([fitness[0] for fitness in random_fitnesses])  # Assuming fitness is a tuple
        plot_ga_vs_random(gen, max_, random_fitnesses, best_set, save_path=CONFIG['plots_path'])
        save_best_baits(best_random_selected_baits, file_path=CONFIG['random_baseline'])
        generate_random_baseline_benchmark(df_norm, CONFIG['number_of_components'], file_path=CONFIG['random_baseline_benchmark'])
        plot_components_correlations_reordered(df_norm, ga_selected_baits, CONFIG['number_of_components'], 'GA', file_path=CONFIG['plots_path'])
        plot_components_correlations_reordered(df_norm, best_random_selected_baits, CONFIG['number_of_components'], 'Random', file_path=CONFIG['plots_path'])
        selected_baits_with_most_preys = baits_with_most_preys(saint_filepath=CONFIG['datasets_path']+'/saint-latest.txt',
                            original_baits_filepath=CONFIG['datasets_path']+'/original_baits.csv',
                            num_baits=60)
        
        plot_components_correlations_reordered(df_norm, selected_baits_with_most_preys, CONFIG['number_of_components'], 'most_preys', file_path=CONFIG['plots_path'])

        basis_matrix_original, basis_matrix_subset, df_subset_reduced, y_original, y_subset = plot_tsne(df_norm, ga_selected_baits, CONFIG['number_of_components'], file_path=CONFIG['plots_path'])
        plot_preys_for_baits(saint_file=CONFIG['datasets_path']+'/saint-latest.txt',
                            #  primary_baits=CONFIG['use_primary_baits'],
                            #  primary_baits_file=CONFIG['datasets_path']+'genesymbols_uniprotids_less_than_110.csv',
                             selected_baits_file=CONFIG['top_features_GA_path']+'/top 1 selected features GA pop500 gen1000.csv',
                             df_norm_file=CONFIG['df_norm_path'],
                             n_components=CONFIG['number_of_components'],
                             file_path=CONFIG['plots_path'])
        perform_gsea_analysis(basis_matrix_original, basis_matrix_subset, df_norm, df_subset_reduced, y_original, y_subset, file_path=CONFIG['gsea_results_path'])




    elif args.step == 'ga_number_of_baits_seeds':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        run_ga_for_number_of_baits_and_seeds(df_norm, CONFIG['number_of_components'], CONFIG['population_size'], CONFIG['number_of_generations'],
                           CONFIG['cxbp'], CONFIG['mutpb'], CONFIG['ga_seeds_path'])
        

    elif args.step == 'seeds_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        get_and_save_top_features_from_ga_seeds(df_norm, CONFIG['ga_seeds_path'], CONFIG['top_features_GA_seeds_path'])
        plot_max_values_for_baits_boxplot_seeds(CONFIG['ga_seeds_path'], CONFIG['plots_path']) 
        # heatmap_number_of_baits()
        plot_nmf_mean_values_with_shades(CONFIG['plots_path'])

    # ML methods
    elif args.step == 'run_ml_methods':
        # Load df_norm
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        all_ml_dfs = run_feature_selection_for_seeds(df_norm, CONFIG['ml_results_path'])

    elif args.step == 'plot_ml_methods':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        all_ml_dfs = pd.read_pickle(CONFIG['ml_results_path']+'all_seeds_ml_results.pkl')

        # List to store results from all seeds
        all_seed_results = []
        methods = ['chi_2','f_classif','mutual_info_classif','lasso','ridge','elastic_net','rf','gbm','xgb', 'nn']  # Assuming df_norm contains method columns

        # Loop through each dataframe
        for idx, ml_df in enumerate(all_ml_dfs):
            # Find best baits using the mean-min correlation function
            best_baits, bait_selections = mean_min_component_correlation_plot_all_methods(
                df_norm, 
                ml_df, 
                CONFIG['number_of_components'], 
                CONFIG['subset_range_ML'], 
                idx+1,
                save_path=f"{CONFIG['plots_path']}/ml_correlation_plot_seed{idx+1}.png"  
            )
            # Add the result of this seed to the list
            all_seed_results.append(bait_selections)

       
            
            # # Plot correlations using best baits
            # plot_components_correlations_reordered(
            #     df_norm, 
            #     best_baits, 
            #     CONFIG['number_of_components'], 
            #     save_path=f"{CONFIG['plots_path']}/ml_component_corr_plot_seed{idx+1}.png"
            # )

        # Calculate and plot the average mean and min across all seeds
        aggregate_mean_min_across_seeds(all_seed_results, CONFIG['subset_range_ML'], methods)
        plot_boxplots_baits_ml(all_seed_results, CONFIG['subset_range_ML'], methods)

        # plot_boxplots_baits_ml()
            
            # # Plot ARI reordered using best baits
            # plot_components_comparison(
            #     df_norm, 
            #     best_baits, 
            #     CONFIG['number_of_components'], 
            #     save_path=f"{CONFIG['plots_path']}/ml_ari_plot_seed{idx+1}.png"
            # )


    elif args.step == 'plot_nmf_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['simulations']) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
    
    elif args.step == 'plot_nmf_scores_min':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_LNCaP_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), CONFIG['simulations']) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
    

        
    elif args.step == 'plot_nmf_cos_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_cos_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6),ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_cos_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['simulations'])#UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS

    elif args.step == 'plot_nmf_cos_scores_min':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_cos_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1),ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_LNCaP_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_cos_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), CONFIG['simulations']) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS


    elif args.step == 'plot_nmf_kl_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_kl_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6),ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0)  #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_kl_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['simulations'])  #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS

    elif args.step == 'plot_nmf_kl_scores_min':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_kl_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1),ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_LNCaP_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_kl_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), CONFIG['simulations']) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS


    elif args.step == 'plot_nmf_ari_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_ari_scores(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1),ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_ari_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['simulations']) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS

    elif args.step == 'plot_nmf_ari_scores_min':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_ari_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_LNCaP_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_ari_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), CONFIG['simulations']) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS


    elif args.step == 'plot_nmf_go_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_go_scores(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1),ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0)  #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_go_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['simulations'])  #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS

    elif args.step == 'plot_nmf_go_scores_min':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_go_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_SW-620_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_nmf_go_scores_min(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), CONFIG['simulations']) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
            
    elif args.step == 'remaining_preys_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_remaining_preys(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_remaining_preys(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['simulations']) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS

    elif args.step == 'leiden_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_leiden_ari(df_norm, CONFIG['leiden_resolutions'],ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_leiden_ari(df_norm) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS

        
    elif args.step == 'gmm_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_gmm_correlation(df_norm, CONFIG['gmm_cluster_numbers'],ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0)#UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_gmm_correlation(df_norm) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS

    elif args.step == 'gmm_hard_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_gmm_hard_ari(df_norm, CONFIG['gmm_cluster_numbers'],ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_gmm_hard_ari(df_norm) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS


    elif args.step == 'go_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_go(df_norm, ga_path=CONFIG['top_features_GA_seeds_path'], ml_path=CONFIG['ml_results_path'], random_path=CONFIG['random_baseline_benchmark'], save_path=CONFIG['plots_path'])
        # df_norm = pd.read_csv('cell_lines/adjusted_prey_bait_matrix_U2-OS_normalized.csv', index_col=0) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS
        # plot_go(df_norm) #UNCOMMENT THIS LINE FOR EVALUATING SIMULATED DATASETS

    elif args.step == 'topology_analysis':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_topology_metrics(df_norm, CONFIG['top_features_GA_path'],
                              CONFIG['random_baseline_benchmark'], CONFIG['ml_results_path'],
                              CONFIG['plots_path'], output_dir=CONFIG['plots_path'])


    elif args.step == 'runtime_analysis':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_runtime_analysis(df_norm, save_path=CONFIG['plots_path'])

    elif args.step == 'individual_components_correlation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_individual_components_vs_bait_sizes(df_norm, save_path=CONFIG['plots_path'])

    elif args.step == 'topology_analysis':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_topology_metrics(df_norm, CONFIG['top_features_GA_path'],
                              CONFIG['random_baseline_benchmark'], CONFIG['ml_results_path'],
                              CONFIG['plots_path'], output_dir=CONFIG['plots_path'])
        
    elif args.step == 'combined_metrics':
        create_combined_metrics_plot(CONFIG['plots_path'])

    elif args.step == 'simulation_expression_analysis': # ONLY FOR DATASET 1
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        get_cellline_data(CONFIG['plots_path'])
        generate_simulated_expression_data(df_norm, CONFIG['cell_lines_path'], CONFIG['plots_path']+'uniprot_mapping.pkl') # ONLY FOR DATASET 1

    elif args.step == 'bait_expression_analysis': # ONLY FOR DATASET 1
        # cell = 'hela'
        plot_baits_expression_heatmap(
            bait_file=f"{CONFIG['top_features_GA_seeds_path']}top_1_features_50_seed_0.csv",
            uniprot_mapping_pickle=CONFIG['plots_path']+'uniprot_mapping.pkl',
            expression_dir=CONFIG['cell_lines_path'],
            output_dir=CONFIG['plots_path']
        )

        plot_nmf_score_vs_expression()

    elif args.step == 'combined_nmf_corr':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(15, 26), range(9, 20), range(14, 25)]
        plot_nmf_correlation_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_scores')

    elif args.step == 'combined_nmf_min_corr':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(15, 26), range(9, 20), range(14, 25)]
        plot_nmf_min_correlation_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_scores_min')

    elif args.step == 'combined_nmf_cos':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(15, 26), range(9, 20), range(14, 25)]
        plot_nmf_cos_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_cos_scores')

    elif args.step == 'combined_nmf_min_cos':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(15, 26), range(9, 20), range(14, 25)]
        plot_nmf_min_cos_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_cos_scores_min')

    elif args.step == 'combined_nmf_kl':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(15, 26), range(9, 20), range(14, 25)]
        plot_nmf_kl_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_kl_scores')

    elif args.step == 'combined_nmf_min_kl':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(15, 26), range(9, 20), range(14, 25)]
        plot_nmf_max_kl_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_kl_scores_max')


    elif args.step == 'combined_nmf_ari':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(20, 21), range(14, 15), range(19, 20)]
        plot_nmf_ari_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_ari_scores')

    elif args.step == 'combined_nmf_min_purity':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(20, 21), range(14, 15), range(19, 20)]
        plot_nmf_min_purity_score_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_purity_scores_min')

    elif args.step == 'combined_nmf_go':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(20, 21), range(14, 15), range(19, 20)]
        plot_nmf_go_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_scores_go')

    elif args.step == 'combined_nmf_min_go':
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(20, 21), range(14, 15), range(19, 20)]
        plot_nmf_min_go_combined(dataset_paths, component_ranges, CONFIG['plots_path'], 'nmf_scores_go_min')


    elif args.step == 'combined_remaining_preys': 
        dataset_paths = [
            'results/dataset1/plots',
            'results/dataset2/plots',
            'results/dataset3/plots'
        ]
        component_ranges = [range(15, 26), range(9, 20), range(14, 25)]
        plot_combined_remaining_preys(dataset_paths, component_ranges, CONFIG['plots_path'], 'remaining_preys')

    elif args.step == 'combined_go_retrieval':
        dataset_paths = [
            'results/dataset1/plots/go_results.pkl',
            'results/dataset2/plots/go_results.pkl',
            'results/dataset3/plots/go_results.pkl'
        ]
        plot_combined_go(dataset_paths, CONFIG['plots_path'], "go_retrieval")

    elif args.step == 'combined_leiden':
        dataset_paths = [
            'results/dataset1/plots/leiden_results.pkl',
            'results/dataset2/plots/leiden_results.pkl',
            'results/dataset3/plots/leiden_results.pkl'
        ]

        plot_leiden_combined(dataset_paths, CONFIG['plots_path'], "leiden_ari")

    elif args.step == 'combined_gmm_hard':
        dataset_paths = [
            'results/dataset1/plots/gmm_hard_results.pkl',
            'results/dataset2/plots/gmm_hard_results.pkl',
            'results/dataset3/plots/gmm_hard_results.pkl'
        ]

        plot_gmm_ari_combined(dataset_paths, CONFIG['plots_path'], "plots/gmm_hard")
   
    elif args.step == 'combined_gmm_correlation':
        dataset_paths = [
            'results/dataset1/plots/gmm_results.pkl',
            'results/dataset2/plots/gmm_results.pkl',
            'results/dataset3/plots/gmm_results.pkl'
        ]
        plot_gmm_combined(dataset_paths, CONFIG['plots_path'], "plots/gmm_correlation")

    elif args.step == 'final_step':
        with open('workflow_completed.log', 'w') as f:
            f.write('Workflow completed successfully')

if __name__ == '__main__':
    main()

