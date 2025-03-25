import yaml

# Load dataset config dynamically
def load_config(dataset):
    with open(f"config/config_{dataset}.yaml", 'r') as file:
        return yaml.safe_load(file)

# Read dataset from Snakemake's `config` instead of `os.getenv`
dataset = config.get("dataset", "dataset1")  # Default is dataset1 if not provided

DATASETS = ["dataset1", "dataset2", "dataset3"]
CONFIGS = {d: load_config(d) for d in DATASETS}

rule all:
    input:
        "workflow_completed_{dataset}.log"

# ====================================
#  STEP 1: LOAD DATA
# ====================================
rule load_data:
    input:
        saint=lambda wildcards: CONFIGS[dataset]["datasets_path"] + "saint-latest.txt",
        # genesymbols=lambda wildcards: CONFIGS[dataset]["datasets_path"] + "genesymbols_uniprotids_less_than_110.csv"
    output:
        CONFIGS[dataset]["df_norm_path"]
    shell:
        "python3 src/main.py --step load_data --config config/config_{dataset}.yaml"

# ====================================
#  STEP 2: RUN GA & EVALUATE RESULTS
# ====================================
rule run_ga:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["ga_results_path"] + "popfile.pkl",
        CONFIGS[dataset]["ga_results_path"] + "logbookfile.pkl",
        CONFIGS[dataset]["ga_results_path"] + "hoffile.pkl"
    shell:
        "python3 src/main.py --step run_ga --config config/config_{dataset}.yaml"

rule ga_evaluation:
    input:
        run_ga_output=rules.run_ga.output
    output:
        CONFIGS[dataset]["plots_path"] + "GA_vs_Random_plot.png",
        CONFIGS[dataset]["plots_path"] + "tsne_original.png"
    shell:
        "python3 src/main.py --step ga_evaluation --config config/config_{dataset}.yaml"

rule ga_number_of_baits_seeds:
    input:
        oad_data_output=rules.load_data.output,
    output:
        pop_files=expand(CONFIGS[dataset]["ga_seeds_path"] + "popfile_features_{num_features}_seed_{seed}.pkl",
                         num_features=range(30, 81), seed=range(10)),
        logbook_files=expand(CONFIGS[dataset]["ga_seeds_path"] + "logbookfile_features_{num_features}_seed_{seed}.pkl",
                             num_features=range(30, 81), seed=range(10)),
        hof_files=expand(CONFIGS[dataset]["ga_seeds_path"] + "hoffile_features_{num_features}_seed_{seed}.pkl",
                         num_features=range(30, 81), seed=range(10))
    shell:
        "python3 src/main.py --step ga_number_of_baits_seeds --config config/config_{dataset}.yaml"

rule seeds_evaluation:
    input:
        ga_number_of_baits_seeds_output=rules.ga_number_of_baits_seeds.output
    output:
        seeds_output_boxplot=CONFIGS[dataset]["plots_path"] + "nbaits vs. max value seeds boxplot GA.png"
    shell:
        "python3 src/main.py --step seeds_evaluation --config config/config_{dataset}.yaml"


# ====================================
#  STEP 3: MACHINE LEARNING FEATURE SELECTION
# ====================================
rule run_ml_methods:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["ml_results_path"] + "all_seeds_ml_results.pkl"
    shell:
        "python3 src/main.py --step run_ml_methods --config config/config_{dataset}.yaml"

rule plot_ml_methods:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nbaits vs. max value seeds boxplot chi_2.png"
    shell:
        "python3 src/main.py --step plot_ml_methods --config config/config_{dataset}.yaml"

# ====================================
#  STEP 4: NMF METRICS RULES
# ====================================
rule plot_nmf_scores:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf scores vs. eachmethod.png"
    shell:
        "python3 src/main.py --step plot_nmf_scores --config config/config_{dataset}.yaml"


rule plot_nmf_scores_min:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf scores vs. each method min.png"
    shell:
        "python3 src/main.py --step plot_nmf_scores_min --config config/config_{dataset}.yaml"


rule plot_nmf_cos_scores:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf cos scores vs. each method.png"
    shell:
        "python3 src/main.py --step plot_nmf_cos_scores --config config/config_{dataset}.yaml"


rule plot_nmf_cos_scores_min:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf cos scores vs. each method min.png"
    shell:
        "python3 src/main.py --step plot_nmf_cos_scores_min --config config/config_{dataset}.yaml"


rule plot_nmf_kl_scores:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf kl scores vs. each method.png"
    shell:
        "python3 src/main.py --step plot_nmf_kl_scores --config config/config_{dataset}.yaml"


rule plot_nmf_kl_scores_min:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf kl scores vs. each method min.png"
    shell:
        "python3 src/main.py --step plot_nmf_kl_scores_min --config config/config_{dataset}.yaml"


rule plot_nmf_ari_scores:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf ari scores vs. each method.png"
    shell:
        "python3 src/main.py --step plot_nmf_ari_scores --config config/config_{dataset}.yaml"


rule plot_nmf_ari_scores_min:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf ari scores vs. each method min.png"
    shell:
        "python3 src/main.py --step plot_nmf_ari_scores_min --config config/config_{dataset}.yaml"



rule plot_nmf_go_scores:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf go components scores vs. each method.png"
    shell:
        "python3 src/main.py --step plot_nmf_go_scores --config config/config_{dataset}.yaml"


rule plot_nmf_go_scores_min:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf go components scores vs. each method min.png"
    shell:
        "python3 src/main.py --step plot_nmf_go_scores_min --config config/config_{dataset}.yaml"




# ====================================
#  STEP 5: NON-NMF METRICS RULES 
# ====================================

rule remaining_preys_evaluation:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "remaining preys vs. each method.png"
    shell:
        "python3 src/main.py --step remaining_preys_evaluation --config config/config_{dataset}.yaml"


rule go_evaluation:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "GO terms retrieval percentage vs. each method (sorted).png"
    shell:
        "python3 src/main.py --step go_evaluation --config config/config_{dataset}.yaml"

rule leiden_evaluation:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "Leiden ARI values vs. each method.png"
    shell:
        "python3 src/main.py --step leiden_evaluation --config config/config_{dataset}.yaml"

rule gmm_hard_evaluation:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "GMM ARI values vs. each method.png"
    shell:
        "python3 src/main.py --step gmm_hard_evaluation --config config/config_{dataset}.yaml"

rule gmm_evaluation:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "GMM mean correlation values vs. each method.png"
    shell:
        "python3 src/main.py --step gmm_evaluation --config config/config_{dataset}.yaml"


rule combined_metrics:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "combined_metrics_comparison_plot_values_nmf_excluded.pdf"

    shell:
        "python3 src/main.py --step combined_metrics --config config/config_{dataset}.yaml"



# ====================================
#  STEP 6: OTHER ANALYSES RULES 
# ====================================
rule topology_analysis:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "degree_distribution_ratio_plot.png"
    shell:
        "python3 src/main.py --step topology_analysis --config config/config_{dataset}.yaml"


rule runtime_analysis:
    input:
        load_data_output = rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "runtime_plot.pdf"
    shell:
        "python3 src/main.py --step runtime_analysis --config config/config_{dataset}.yaml"


rule individual_components_correlation:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_component_bait_size_comparison.pdf"

    shell:
        "python3 src/main.py --step individual_components_correlation --config config/config_{dataset}.yaml"


# ====================================
#  STEP 7: DATASET1-SPECIFIC STEPS
# ====================================
rule bait_expression_analysis:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS["dataset1"]["plots_path"] + "bait_expression_analysis.png"
    shell:
        "python3 src/main.py --step bait_expression_analysis --config config/config_dataset1.yaml"

rule simulation_expression_analysis:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS["dataset1"]["plots_path"] + "simulation_expression_analysis.png"
    shell:
        "python3 src/main.py --step simulation_expression_analysis --config config/config_dataset1.yaml"


### RUN THIS AFTER RUNNING THE PREVIOUS RULES FOR ALL OTHER DATASETS ###
# ====================================
#  STEP 8: COMBINED DATASETS PLOTS
# ====================================
rule combined_nmf_corr:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_corr --config config/config_{dataset}.yaml"

rule combined_nmf_min_corr:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_min_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_min_corr --config config/config_{dataset}.yaml"

rule combined_nmf_cos:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_cos_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_cos --config config/config_{dataset}.yaml"

rule combined_nmf_min_cos:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_cos_min_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_min_cos --config config/config_{dataset}.yaml"

rule combined_nmf_kl:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_kl_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_kl --config config/config_{dataset}.yaml"

rule combined_nmf_min_kl:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_kl_min_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_min_kl --config config/config_{dataset}.yaml"

rule combined_nmf_ari:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_ari_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_ari --config config/config_{dataset}.yaml"

rule combined_nmf_min_purity:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_ari_min_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_min_purity --config config/config_{dataset}.yaml"

rule combined_nmf_go:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_go_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_go --config config/config_{dataset}.yaml"

rule combined_nmf_min_go:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_go_min_comparison.png"
    shell:
        "python3 src/main.py --step combined_nmf_min_go --config config/config_{dataset}.yaml"

rule combined_remaining_preys:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "remaining_preys_comparison.png"
    shell:
        "python3 src/main.py --step combined_remaining_preys --config config/config_{dataset}.yaml"

rule combined_go_retrieval:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "GO_retrieval_percentage_comparison.png"
    shell:
        "python3 src/main.py --step combined_go_retrieval --config config/config_{dataset}.yaml"

rule combined_leiden:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "leiden_comparison.png"
    shell:
        "python3 src/main.py --step combined_leiden --config config/config_{dataset}.yaml"

rule combined_gmm_hard:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "gmm_hard_comparison.png"
    shell:
        "python3 src/main.py --step combined_gmm_hard --config config/config_{dataset}.yaml"

rule combined_gmm_correlation:
    input:
        load_data_output=rules.load_data.output,
    output:
        CONFIGS[dataset]["plots_path"] + "gmm_correlation_comparison.png"
    shell:
        "python3 src/main.py --step combined_gmm_correlation --config config/config_{dataset}.yaml"


# ====================================
#  FINAL STEP: MARK WORKFLOW AS COMPLETED
# ====================================
rule finalize_workflow:
    input:
        CONFIGS[dataset]["plots_path"] + "nmf_scores_vs_each_method.png"
    output:
        "workflow_completed_{dataset}.log"
    shell:
        "echo 'Workflow completed for {dataset}' > {output}"
