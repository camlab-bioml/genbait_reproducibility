# GENBAIT Reproducibility

This repository contains the GENBAIT project for bait selection in BioID experiments. This project is designed to be reproducible using Snakemake. Below are the instructions on how to reproduce the results of each step in the workflow using the provided configuration files.

## Requirements

Before running the workflow, ensure you have the following installed:

- [Python 3.6+](https://www.python.org/downloads/)
- [Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)
- [Git LFS](https://git-lfs.github.com/) (for handling large files)

## Setup

### Create a Virtual Environment

It is recommended to create a virtual environment to manage dependencies:

```sh
python -m venv genbait_env
source genbait_env/bin/activate  # On Windows use `genbait_env\Scripts\activate`
```

### Install the Package
Navigate to the root directory of the project and run:
```sh
git clone https://github.com/vesalkasmaeifar/genbait_reproducibility.git
cd genbait_reproducibility
pip install .
```
This will install the package along with all required dependencies.


## Running the Workflow
You can reproduce the results for each dataset by running the Snakemake workflow. The configuration files for dataset1 and dataset2 are provided in the config directory.


### Load data
```sh
snakemake --cores 1 data/dataset1/df_norm.csv --configfile config/config_dataset1.yaml
snakemake --cores 1 data/dataset2/df_norm.csv --configfile config/config_dataset2.yaml
```

### Run genetic algorithm
```sh
snakemake --cores 1 results/dataset1/GA_results/popfile.pkl --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/GA_results/popfile.pkl --configfile config/config_dataset2.yaml
```

### Get genetic algorithm results
```sh
snakemake --cores 1 results/dataset1/plots/GA_vs_Random_plot.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/GA_vs_Random_plot.png --configfile config/config_dataset2.yaml
```

### Run genetic algorithm with different number of baits and seeds
```sh
snakemake --cores 1 results/dataset1/GA_number_of_baits_seeds/popfile_features_{num_features}_seed_{seed}.pkl --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/GA_number_of_baits_seeds/popfile_features_{num_features}_seed_{seed}.pkl --configfile config/config_dataset2.yaml
```

### Visualize results for different number of baits and seeds
```sh
snakemake --cores 1 results/dataset1/plots/nbaits_vs_max_value_seeds_boxplot_GA.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/nbaits_vs_max_value_seeds_boxplot_GA.png --configfile config/config_dataset2.yaml
```

### Run machine learning methods
```sh
snakemake --cores 1 results/dataset1/plots/ml_correlation_plot_averaged.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/ml_correlation_plot_averaged.png --configfile config/config_dataset2.yaml
```

### Plot NMF scores
```sh
snakemake --cores 1 results/dataset1/plots/nmf_scores_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/nmf_scores_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot NMF Cosine similarity scores
```sh
snakemake --cores 1 results/dataset1/plots/nmf_cos_scores_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/nmf_cos_scores_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot NMF KL Divergence scores
```sh
snakemake --cores 1 results/dataset1/plots/nmf_kl_scores_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/nmf_kl_scores_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot NMF ARI
```sh
snakemake --cores 1 results/dataset1/plots/nmf_ari_values_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/nmf_ari_values_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot NMF GO Jaccard index
```sh
snakemake --cores 1 results/dataset1/plots/nmf_go_components_scores_values_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/nmf_go_components_scores_values_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot remaining preys percentage
```sh
snakemake --cores 1 results/dataset1/plots/remaining_preys_vs_each_method_sorted.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/remaining_preys_vs_each_method_sorted.png --configfile config/config_dataset2.yaml
```

### Plot GO retrieval percentage
```sh
snakemake --cores 1 results/dataset1/plots/GO_terms_retrieval_percentage_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/GO_terms_retrieval_percentage_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot Leiden ARI 
```sh
snakemake --cores 1 results/dataset1/plots/Leiden_ARI_values_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/Leiden_ARI_values_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot GMM ARI
```sh
snakemake --cores 1 results/dataset1/plots/GMM_ARI_values_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/GMM_ARI_values_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot GMM mean correlation
```sh
snakemake --cores 1 results/dataset1/plots/GMM_mean_correlation_values_vs_each_method.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/GMM_mean_correlation_values_vs_each_method.png --configfile config/config_dataset2.yaml
```

### Plot combined metrics
```sh
snakemake --cores 1 results/dataset1/plots/combined_metrics_comparison_plot.png --configfile config/config_dataset1.yaml
snakemake --cores 1 results/dataset2/plots/combined_metrics_comparison_plot.png --configfile config/config_dataset2.yaml
```

