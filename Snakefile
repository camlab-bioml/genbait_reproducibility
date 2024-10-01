import glob

DATASETS = ["dataset1", "dataset2"]

rule all:
    input:
        expand("workflow_completed_{dataset}.log", dataset=DATASETS),

rule load_data:
    input:
        saint="data/{dataset}/saint-latest.txt",
        genesymbols="data/{dataset}/genesymbols_uniprotids_less_than_110.csv"
    output:
        "data/{dataset}/df_norm.csv"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step load_data --config config/config_{params.dataset}.yaml"

rule run_ga:
    input:
        df_norm="data/{dataset}/df_norm.csv"
    output:
        pop_file="results/{dataset}/GA_results/popfile.pkl",
        logbook_file="results/{dataset}/GA_results/logbookfile.pkl",
        hof_file="results/{dataset}/GA_results/hoffile.pkl"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step run_ga --config config/config_{params.dataset}.yaml"

rule ga_evaluation:
    input:
        run_ga_output=rules.run_ga.output
    output:
        top_features_ga=expand("results/{dataset}/top_features_GA/top {number} selected features GA pop500 gen1000.csv", number=range(1, 11)),
        plot_ga_vs_random="results/{dataset}/plots/GA_vs_Random_plot.png",
        plot_components_correlation="results/{dataset}/plots/GA_component_corr_plot.png",
        plot_lost_preys="results/{dataset}/plots/Lost_Preys_Plot.png",
        tsne_plot="results/{dataset}/plots/tsne_original.png",
        tsne_plot_subset='results/{dataset}/plots/tsne_subset.png',
        tsne_plot_baits='results/{dataset}/plots/preys_for_each_bait.png',
        gsea_results_original="results/{dataset}/gsea_results/GSEA_basis_original.xlsx",
        gsea_results_subset="results/{dataset}/gsea_results/GSEA_basis_subset.xlsx"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step ga_evaluation --config config/config_{params.dataset}.yaml"

# rule penalty_evaluation:
#     input:
#         df_norm="data/{dataset}/df_norm.csv"
#     output:
#         penalty_evaluation_output="results/{dataset}/plots/fitness_penalty_diagonal_mean.png"
#     params:
#         dataset="{dataset}"
#     shell:
#         "python3 src/main.py --step penalty_evaluation --config config/config_{params.dataset}.yaml"

# rule penalty_evaluation_range:
#     input:
#         df_norm="data/{dataset}/df_norm.csv"
#     output:
#         penalty_evaluation_output=expand("results/{dataset}/plots/fitness_penalty_diagonal_mean_{baits}.png", baits=range(56, 61))
#     params:
#         dataset="{dataset}"
#     shell:
#         "python3 src/main.py --step penalty_evaluation_range --config config/config_{params.dataset}.yaml"

# rule ga_number_of_baits:
#     input:
#         df_norm="data/{dataset}/df_norm.csv"
#     output:
#         pop_files=expand("results/{dataset}/GA_number_of_baits/popfile{num_features}.pkl", num_features=range(30, 81)),
#         logbook_files=expand("results/{dataset}/GA_number_of_baits/logbookfile{num_features}.pkl", num_features=range(30, 81)),
#         hof_files=expand("results/{dataset}/GA_number_of_baits/hoffile{num_features}.pkl", num_features=range(30, 81))
#     params:
#         dataset="{dataset}"
#     shell:
#         "python3 src/main.py --step ga_number_of_baits --config config/config_{params.dataset}.yaml"

rule ga_number_of_baits_seeds:
    input:
        df_norm="data/{dataset}/df_norm.csv"
    output:
        pop_files=expand("results/{dataset}/GA_number_of_baits_seeds/popfile_features_{num_features}_seed_{seed}.pkl", num_features=range(30, 81), seed=range(10)),
        logbook_files=expand("results/{dataset}/GA_number_of_baits_seeds/logbookfile_features_{num_features}_seed_{seed}.pkl", num_features=range(30, 81), seed=range(10)),
        hof_files=expand("results/{dataset}/GA_number_of_baits_seeds/hoffile_features_{num_features}_seed_{seed}.pkl", num_features=range(30, 81), seed=range(10))
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step ga_number_of_baits_seeds --config config/config_{params.dataset}.yaml"

# rule nbaits_evaluation:
#     input:
#         ga_number_of_baits_output=rules.ga_number_of_baits.output
#     output:
#         nbaits_output="results/{dataset}/plots/nbaits_vs_max_value.png",
#         nbaits_output_boxplot="results/{dataset}/plots/nbaits_vs_max_value_boxplot.png"
#     params:
#         dataset="{dataset}"
#     shell:
#         "python3 src/main.py --step nbaits_evaluation --config config/config_{params.dataset}.yaml"

rule seeds_evaluation:
    input:
        ga_number_of_baits_seeds_output=rules.ga_number_of_baits_seeds.output
    output:
        seeds_output_boxplot="results/{dataset}/plots/nbaits_vs_max_value_seeds_boxplot_GA.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step seeds_evaluation --config config/config_{params.dataset}.yaml"

rule ml_methods:
    input:
        df_norm="data/{dataset}/df_norm.csv"
    output:
        "results/{dataset}/plots/ml_correlation_plot_averaged.png",
        expand(
            ["results/{dataset}/plots/ml_correlation_plot_seed{seed}.png",
             "results/{dataset}/plots/ml_component_corr_plot_seed{seed}.png"],
            seed=range(1, 11)
        )
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step ml_methods --config config/config_{params.dataset}.yaml"

rule plot_nmf_scores:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        ga_nmf_plot="results/{dataset}/plots/nmf_scores_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step plot_nmf_scores --config config/config_{params.dataset}.yaml"

rule plot_nmf_ari_scores:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        ga_nmf_ari_plot="results/{dataset}/plots/nmf_ari_values_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step plot_nmf_ari_scores --config config/config_{params.dataset}.yaml"

rule plot_nmf_cos_scores:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        ga_nmf_cos_plot="results/{dataset}/plots/nmf_cos_scores_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step plot_nmf_cos_scores --config config/config_{params.dataset}.yaml"

rule plot_nmf_kl_scores:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        ga_nmf_kl_plot="results/{dataset}/plots/nmf_kl_scores_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step plot_nmf_kl_scores --config config/config_{params.dataset}.yaml"

rule plot_nmf_spearman_scores:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        ga_nmf_spearman_plot="results/{dataset}/plots/nmf_spearman_scores_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step plot_nmf_spearman_scores --config config/config_{params.dataset}.yaml"

rule plot_nmf_go_scores:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        ga_nmf_go_plot="results/{dataset}/plots/nmf_go_components_scores_values_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step plot_nmf_go_scores --config config/config_{params.dataset}.yaml"

rule plot_remaining_preys:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        ga_remaining_preys_plot="results/{dataset}/plots/remaining_preys_vs_each_method_sorted.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step plot_remaining_preys --config config/config_{params.dataset}.yaml"

# rule plot_error_rate:
#     input:
#         nmf_scores_output=rules.plot_nmf_scores.output
#     output:
#         error_rate_output="results/{dataset}/plots/nmf_error_rate.png"
#     params:
#         dataset="{dataset}"
#     shell:
#         "python3 src/main.py --step plot_error_rate --config config/config_{params.dataset}.yaml"

rule leiden_evaluation:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        "results/{dataset}/plots/Leiden_ARI_values_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step leiden_evaluation --config config/config_{params.dataset}.yaml"

rule gmm_evaluation:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        "results/{dataset}/plots/GMM_mean_correlation_values_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step gmm_evaluation --config config/config_{params.dataset}.yaml"

rule gmm_hard_evaluation:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        "results/{dataset}/plots/GMM_ARI_values_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step gmm_hard_evaluation --config config/config_{params.dataset}.yaml"

rule go_evaluation:
    input:
        ga_evaluation_output=rules.ga_evaluation.output,
        ml_methods_output=rules.ml_methods.output
    output:
        "results/{dataset}/plots/GO_terms_retrieval_percentage_vs_each_method.png"
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step go_evaluation --config config/config_{params.dataset}.yaml"

rule combined_metrics:
    input:
        leiden_output="results/{dataset}/plots/Leiden_ARI_values_vs_each_method.png",
        gmm_output="results/{dataset}/plots/GMM_mean_correlation_values_vs_each_method.png",
        gmm_hard_output="results/{dataset}/plots/GMM_ARI_values_vs_each_method.png",
        go_output="results/{dataset}/plots/GO_terms_retrieval_percentage_vs_each_method.png"
    output:
        'results/{dataset}/plots/combined_metrics_comparison_plot.png'
    params:
        dataset="{dataset}"
    shell:
        "python3 src/main.py --step combined_metrics --config config/config_{params.dataset}.yaml"

rule finalize_workflow:
    input:
        combined_metrics_output='results/{dataset}/plots/combined_metrics_comparison_plot.png'
    output:
        "workflow_completed_{dataset}.log"
    params:
        dataset="{dataset}"
    shell:
        "echo 'Workflow completed for {wildcards.dataset}' > {output}"
