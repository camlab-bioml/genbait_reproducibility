import argparse
import pandas as pd
import os
from data_loading import load_data, preprocess_data
from genetic_algorithm import run_genetic_algorithm
from genetic_algorithm import evalSubsetCorrelation
from genetic_algorithm_number_of_baits import run_ga_for_numbers
from data_storage import save_genetic_algorithm_results, load_genetic_algorithm_results
from random_baseline import generate_random_baseline, generate_random_baseline_benchmark, generate_best_set_sequence_from_random, save_best_baits
from plot_GA_vs_random import plot_ga_vs_random
from save_top_features import get_and_save_top_features_from_ga, get_and_save_top_features_from_ga_seeds
from plot_components_correlation import plot_components_correlations_reordered
from plot_components_comparison import plot_components_comparison
from plot_lost_prey import plot_lost_preys_from_features
from plot_tsne import plot_tsne
from gsea_analysis import perform_gsea_analysis
from ml_feature_selection_methods import run_feature_selection_for_seeds
from plot_ml_methods import mean_min_component_correlation_plot_all_methods
from plot_ml_methods import aggregate_mean_min_across_seeds
from plot_ml_methods import plot_boxplots_baits_ml, plot_boxplots_baits_ml_pkl
from plot_combined_metrics import create_combined_metrics_plot
from plot_nbaits_vs_maxvalue import plot_max_values_for_baits, plot_max_values_for_baits_boxplot
from plot_nbaits_vs_maxvalue_seeds import plot_max_values_for_baits_boxplot_seeds, heatmap_number_of_baits
from genetic_algorithm_number_of_baits_seeds import run_ga_for_number_of_baits_and_seeds
from genetic_algorithm_manual import run_genetic_algorithm_manual, run_genetic_algorithm_manual_range
from plot_fitness_penalty_GA import plot_fitness_diagonal_mean_penalty, plot_fitness_diagonal_mean_penalty_range
from plot_error_rate_nmf import plot_error_rate_nmf
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
from plot_nmf_spearman_scores import plot_nmf_spearman_scores
from plot_nmf_go_component import plot_nmf_go_scores
from plot_mean_shaded_scores_vs_nbaits import plot_nmf_mean_values_with_shades
from plot_baits_jaccard_index import plot_seed_comparison_heatmaps
from deap import base, creator, tools
import yaml
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_directories_exist(config):
    directories = [
        config["ga_results_path"],
        config["ml_results_path"],
        config["plots_path"],
        config["fitness_penalty_path"],
        config["top_features_GA_path"],
        config["top_features_ML_path"],
        config["gsea_results_path"],
        config["random_baseline"],
        config["random_baseline_benchmark"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    
    parser = argparse.ArgumentParser(description='Bait Selection for BioID Map')
    parser.add_argument('--step', choices=['load_data', 'run_ga', 'ga_evaluation', 'penalty_evaluation', 'penalty_evaluation_range', 'ga_number_of_baits', 'ga_number_of_baits_seeds', 'nbaits_evaluation', 'seeds_evaluation', 'ml_methods', 'plot_nmf_scores', 'plot_nmf_ari_scores', 'plot_nmf_cos_scores', 'plot_nmf_kl_scores', 'plot_nmf_spearman_scores', 'plot_nmf_go_scores', 'plot_remaining_preys', 'leiden_evaluation', 'gmm_evaluation', 'gmm_hard_evaluation', 'go_evaluation', 'combined_metrics',  'final_step'], help='Which step to execute')
    parser.add_argument('--config', required=True, help='Path to the configuration file')
    args = parser.parse_args()
    CONFIG = load_config(args.config)
    ensure_directories_exist(CONFIG)


    # Load Data
    if args.step == 'load_data':
        df = load_data(CONFIG['datasets_path'] + 'saint-latest.txt', sep='\t')
        
        if CONFIG.get('use_primary_baits', False):  # Checks if use_primary_baits is True
            primary_baits_df = load_data(CONFIG['datasets_path'] + '/genesymbols_uniprotids_less_than_110.csv', index_col=0)
            primary_baits = list(primary_baits_df['Gene Symbols'])
            df_norm = preprocess_data(df, primary_baits, file_path=CONFIG['datasets_path'])
        else:
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

        
    # Evaluation and Plotting
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
        
        basis_matrix_original, basis_matrix_subset, df_subset_reduced, y_original, y_subset = plot_tsne(df_norm, ga_selected_baits, CONFIG['number_of_components'], file_path=CONFIG['plots_path'])
        plot_preys_for_baits(saint_file=CONFIG['datasets_path']+'saint-latest.txt',
                             primary_baits=CONFIG['use_primary_baits'],
                             primary_baits_file=CONFIG['datasets_path']+'genesymbols_uniprotids_less_than_110.csv',
                             selected_baits_file=CONFIG['top_features_GA_path']+'top 1 selected features GA pop500 gen1000.csv',
                             df_norm_file=CONFIG['df_norm_path'],
                             n_components=CONFIG['number_of_components'],
                             file_path=CONFIG['plots_path'])
        perform_gsea_analysis(basis_matrix_original, basis_matrix_subset, df_norm, df_subset_reduced, y_original, y_subset, file_path=CONFIG['gsea_results_path'])

    # elif args.step == 'penalty_evaluation':
    #     df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
    #     run_genetic_algorithm_manual(df_norm,
    #                         n_components=CONFIG["number_of_components"],
    #                         subset_range=CONFIG["subset_range_GA"],
    #                         population_size=CONFIG["population_size"],
    #                         n_generations=CONFIG["number_of_generations"],
    #                         cxpb=CONFIG["cxbp"],
    #                         mutpb=CONFIG["mutpb"])
        
    #     plot_fitness_diagonal_mean_penalty()

        
    # elif args.step == 'penalty_evaluation_range':
    #     df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
    #     run_genetic_algorithm_manual_range(df_norm,
    #                         n_components=CONFIG["number_of_components"],
    #                         population_size=CONFIG["population_size"],
    #                         n_generations=CONFIG["number_of_generations"],
    #                         cxpb=CONFIG["cxbp"],
    #                         mutpb=CONFIG["mutpb"])
        
    #     plot_fitness_diagonal_mean_penalty_range()


    # elif args.step == 'ga_number_of_baits':
    #     df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
    #     run_ga_for_numbers(df_norm, CONFIG['number_of_components'], CONFIG['population_size'], CONFIG['number_of_generations'],
    #                        CONFIG['cxbp'], CONFIG['mutpb'])
        


    elif args.step == 'ga_number_of_baits_seeds':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        run_ga_for_number_of_baits_and_seeds(df_norm, CONFIG['number_of_components'], CONFIG['population_size'], CONFIG['number_of_generations'],
                           CONFIG['cxbp'], CONFIG['mutpb'], file_path=CONFIG['ga_seeds_path'])
        

    # elif args.step == 'nbaits_evaluation':
    #     plot_max_values_for_baits()
    #     plot_max_values_for_baits_boxplot()

    elif args.step == 'seeds_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        get_and_save_top_features_from_ga_seeds(df_norm,
                                                ga_dir=CONFIG['ga_seeds_path'], 
                                                save_dir=CONFIG['top_features_GA_seeds_path'])
        plot_max_values_for_baits_boxplot_seeds(ga_seeds_path=CONFIG['ga_seeds_path'],
                                                plots_path=CONFIG['plots_path']) 
        plot_nmf_mean_values_with_shades(plots_path=CONFIG['plots_path'])
        plot_seed_comparison_heatmaps(CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])

    # ML methods
    elif args.step == 'ml_methods':
         # Load df_norm
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        all_ml_dfs = run_feature_selection_for_seeds(df_norm, CONFIG['ml_results_path'])

        # List to store results from all seeds
        all_seed_results = []
        methods = ['chi_2','f_classif','mutual_info_classif','lasso','ridge','elastic_net','rf','gbm','xgb']  # Assuming df_norm contains method columns

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

       
            
            # Plot correlations using best baits
            plot_components_correlations_reordered(
                df_norm, 
                best_baits, 
                CONFIG['number_of_components'], 
                save_path=f"{CONFIG['plots_path']}/ml_component_corr_plot_seed{idx+1}.png"
            )

        # Calculate and plot the average mean and min across all seeds
        aggregate_mean_min_across_seeds(all_seed_results, CONFIG['subset_range_ML'], methods)
        plot_boxplots_baits_ml(all_seed_results, CONFIG['subset_range_ML'], methods)

        plot_boxplots_baits_ml_pkl(plots_path=CONFIG['plots_path'])
            
            # # Plot ARI reordered using best baits
            # plot_components_comparison(
            #     df_norm, 
            #     best_baits, 
            #     CONFIG['number_of_components'], 
            #     save_path=f"{CONFIG['plots_path']}/ml_ari_plot_seed{idx+1}.png"
            # )

   

    elif args.step == 'plot_nmf_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])
    
    elif args.step == 'plot_nmf_ari_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_ari_scores(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])

    elif args.step == 'plot_nmf_cos_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_cos_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])
    
    elif args.step == 'plot_nmf_kl_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_kl_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])

    # elif args.step == 'plot_nmf_spearman_scores':
    #     df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
    #     plot_nmf_spearman_scores(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6))
            

    elif args.step == 'plot_nmf_go_scores':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_nmf_go_scores(df_norm, range(CONFIG['number_of_components'], CONFIG['number_of_components']+1), CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])
            

    elif args.step == 'plot_remaining_preys':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_remaining_preys(df_norm, range(CONFIG['number_of_components']-5, CONFIG['number_of_components']+6), CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])


    elif args.step == 'leiden_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_leiden_ari(df_norm, CONFIG['leiden_cluster_numbers'], CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])

        
    elif args.step == 'gmm_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_gmm_correlation(df_norm, CONFIG['gmm_cluster_numbers'] ,CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])

    elif args.step == 'gmm_hard_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_gmm_hard_ari(df_norm, CONFIG['gmm_cluster_numbers'] ,CONFIG['top_features_GA_path'], CONFIG['ml_results_path'], CONFIG['random_baseline_benchmark'], CONFIG['plots_path'])


    elif args.step == 'go_evaluation':
        df_norm = pd.read_csv(CONFIG["df_norm_path"], index_col=0)
        plot_go(df_norm)
        
    elif args.step == 'combined_metrics':
        create_combined_metrics_plot(CONFIG['plots_path'])

   

    elif args.step == 'final_step':
        with open('workflow_completed.log', 'w') as f:
            f.write('Workflow completed successfully')

if __name__ == '__main__':
    main()

