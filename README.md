# GENBAIT Reproducibility

This repository contains the results of all analyses in the GENBAIT project for bait selection in BioID experiments. 

A **preprint** describing the method and introducing a novel benchmarking platform is available:  
[Kasmaeifar et al. (2024) _Computational design and evaluation of optimal bait sets for scalable proximity proteomics_](https://www.biorxiv.org/content/10.1101/2024.10.03.616533v1)

This project is designed to be reproducible using Snakemake. Below are the instructions on how to reproduce the results of each step in the workflow using the provided configuration files.

---

## Requirements

Before running the workflow, ensure you have the following installed:

- [Python 3.10+](https://www.python.org/downloads/)
- [Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)
- [Git LFS](https://git-lfs.github.com/)

---

## Build Tools Installation

Some packages used in this repository (e.g., `shap`, `xgboost`, `leidenalg`) require compilation and system build tools.

### For Windows

1. **Download Microsoft C++ Build Tools**:  
   [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

2. **In the installer, select the following**:
   - **C++ build tools** workload
   - **MSVC v14 or later**
   - **Windows 10 or 11 SDK**

3. Complete the installation and **restart your terminal**.

> If these tools are missing, you may encounter errors such as:  
> `error: Microsoft Visual C++ 14.0 or greater is required`

---

### For macOS

1. **Install Xcode Command Line Tools**:  
   Open Terminal and run:
   ```bash
   xcode-select --install


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
git clone https://github.com/camlab-bioml/genbait_reproducibility.git
cd genbait_reproducibility
pip install .
```
This will install the package along with all required dependencies.


## Running the Workflow
To reproduce the results for each dataset, run the Snakemake workflow. The configuration files for each dataset are located in the `config/` directory.

```sh
# Example: Load data step for dataset1 using all available CPU cores
snakemake --cores all load_data --config dataset=dataset1

# 1. Load data
snakemake --cores 1 load_data


# 2. GENBAIT evaluation

# Run the Genetic Algorithm (GA)
snakemake --cores 1 run_ga

# Evaluation
snakemake --cores 1 ga_evaluation

# Run GENBAIT for different bait lenghts and seeds
snakemake --cores 1 ga_number_of_baits_seeds

# Bait lengths and seeds evaluation
snakemake --cores 1 seeds_evaluation


# 3. Machine learning feature selection

# Run ML Methods
snakemake --cores 1 run_ml_methods

# Plot ML Methods
snakemake --cores 1 plot_ml_methods


# 4. NMF metrics rules

# Mean NMF correlation
snakemake --cores 1 plot_nmf_scores

# Min NMF correlation
snakemake --cores 1 plot_nmf_scores_min

# Mean NMF Cosine similarity
snakemake --cores 1 plot_nmf_cos_scores

# Min NMF Cosine similarity
snakemake --cores 1 plot_nmf_cos_scores_min

# Mean NMF KL divergence
snakemake --cores 1 plot_nmf_kl_scores

# Max NMF KL divergence
snakemake --cores 1 plot_nmf_kl_scores_min

# NMF ARI
snakemake --cores 1 plot_nmf_ari_scores

# Min NMF purity score
snakemake --cores 1 plot_nmf_ari_scores_min

# Mean NMF Jaccard GO index
snakemake --cores 1 plot_nmf_go_scores

# Min NMF Jaccard GO index
snakemake --cores 1 plot_nmf_go_scores_min


# 5. Non-NMF metrics rules

# Remaining preys percentage
snakemake --cores 1 remaining_preys_evaluation

# GO retrieval percentage
snakemake --cores 1 go_evaluation

# Leiden ARI
snakemake --cores 1 leiden_evaluation

# GMM ARI
snakemake --cores 1 gmm_hard_evaluation

# Mean GMM correlation
snakemake --cores 1 gmm_evaluation

# Combined metrics plot
snakemake --cores 1 combined_metrics


# 6. Other analyses rules 

# Topology analysis
snakemake --cores 1 topology_analysis

# Runtime analysis
snakemake --cores 1 runtime_analysis

# Individual components correlation
snakemake --cores 1 individual_components_correlation


# 7. Dataset1-specific analysis

# Bait expression analysis
snakemake --cores 1 bait_expression_analysis

# Simulation expression analysis
snakemake --cores 1 simulation_expression_analysis

# 8. Cobined datasets plots
# Comined mean NMF correlation
snakemake --cores 1 combined_nmf_corr

# Comined min NMF correlation
snakemake --cores 1 combined_nmf_min_corr

# Comined mean NMF Cosine similarity
snakemake --cores 1 combined_nmf_cos

# Comined min NMF Cosine similarity
snakemake --cores 1 combined_nmf_min_cos

# Comined mean NMF KL divergence
snakemake --cores 1 combined_nmf_kl

# Comined min NMF KL divergence
snakemake --cores 1 combined_nmf_min_kl

# Comined NMF ARI
snakemake --cores 1 combined_nmf_ari

# Comined min NMF purity score
snakemake --cores 1 combined_nmf_min_purity

# Comined mean NMF go
snakemake --cores 1 combined_nmf_go

# Comined min NMF go
snakemake --cores 1 combined_nmf_min_go

# Combined remaining preys
snakemake --cores 1 combined_remaining_preys

# Combined GO retrieval 
snakemake --cores 1 combined_go_retrieval

# Combined Leiden ARI
snakemake --cores 1 combined_leiden

# Combined GMM ARI
snakemake --cores 1 combined_gmm_hard

# Combined mean GMM correlation
snakemake --cores 1 combined_gmm_correlation

# 9. Final step: mark workflow as completed
snakemake --cores 1 finalize_workflow
