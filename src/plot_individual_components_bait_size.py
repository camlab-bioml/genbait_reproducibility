import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from gprofiler import GProfiler
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def compute_nmf_component_correlations(df_norm, selected_baits, number_of_components):
    """
    Computes Pearson correlation for each NMF component between full and subset dataset.

    Parameters:
    - df_norm: DataFrame containing the full prey-bait matrix (preys = columns, baits = rows).
    - selected_baits: List of selected baits.
    - number_of_components: Number of NMF components.

    Returns:
    - List of Pearson correlations (one per NMF component).
    """
    
    # Convert dataframes to numpy arrays
    original_data = df_norm.to_numpy()
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    subset_data = original_data[subset_indices, :]

    # Decomposition with NMF
    nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_original = nmf.fit_transform(original_data)
    basis_matrix_original = nmf.components_.T

    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Reorder basis matrix of the subset using linear sum assignment
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    _, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

     # Calculating correlation matrix
    corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:number_of_components, number_of_components:]
    
    return np.diag(corr_matrix)  # Returns a list of correlations for all components


def plot_individual_components_vs_bait_sizes(df_norm, save_path):

    # Load selected baits file
    bait_selection_df = pd.read_csv(f"{save_path}runtime_analysis_selected_baits_repeat_1.csv")

    # Define bait lengths and methods
    bait_lengths = [30, 60, 90]
    methods = ["Chi2", "F_classif", "Mutual_Info", "Lasso", "Ridge", "Elastic_Net", "Random_Forest",
            "Gradient_Boosting", "XGBoost", "Neural_Network", "GENBAIT"]

    # Set number of NMF components
    number_of_components = 20  # Adjust based on previous findings

    # Store detailed correlation results
    correlation_results = []

    # Loop through bait lengths and methods
    for length in bait_lengths:
        for method in methods:
            column_name = f"{method}_{length}"
            selected_baits = bait_selection_df[column_name].dropna().tolist()
            component_correlations = compute_nmf_component_correlations(df_norm, selected_baits, number_of_components)
            
            # Store results for each component
            for component_idx, correlation_value in enumerate(component_correlations):
                correlation_results.append({
                    "Method": method,
                    "Bait_Length": length,
                    "Component": component_idx + 1,  # Component index (1-based)
                    "NMF_Correlation": correlation_value
                })

    # Convert to DataFrame
    correlation_df = pd.DataFrame(correlation_results)

    # Save results for later use
    correlation_df.to_csv(f"{save_path}nmf_component_bait_size_comparison.csv", index=False)


    plt.figure(figsize=(12, 8))
    heatmap_data = correlation_df.pivot_table(index="Component", columns=["Method", "Bait_Length"], values="NMF_Correlation")
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=False, vmin=-1, vmax=1, cbar=True)
    plt.title("NMF Component Correlations Across Methods and Bait Lengths")
    plt.xlabel("Method and Bait Length")
    plt.ylabel("NMF Component")
    plt.savefig(f'{save_path}nmf_component_bait_size_comparison.pdf')
    plt.savefig(f'{save_path}nmf_component_bait_size_comparison.png')
    # plt.show()


    ### **Step 1: Save Low-Correlation Components**
    low_correlation_df = correlation_df[correlation_df["NMF_Correlation"] < 0.50]
    low_correlation_df.to_csv(f"{save_path}nmf_component_bait_size_low_correlation_components.csv", index=False)

    ### **Step 2: Assign Preys to Their Primary NMF Component**
    def assign_preys_to_primary_component(df_norm, basis_matrix):
        """
        Assigns each prey to its primary NMF component based on the highest value in the basis matrix.

        Parameters:
        - df_norm: DataFrame with original prey-bait matrix (preys = columns).
        - basis_matrix: NMF basis matrix (preys x components).

        Returns:
        - DataFrame where each prey is assigned to its strongest NMF component.
        """
        prey_names = df_norm.columns  # Preys are the column names in df_norm
        primary_components = np.argmax(basis_matrix, axis=1) + 1  # Find component with highest value for each prey

        # Create DataFrame mapping preys to their strongest component
        prey_component_df = pd.DataFrame({
            "Prey": prey_names,
            "Primary_Component": primary_components
        })

        return prey_component_df

    # Run NMF on the full dataset
    nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
    W_full = nmf.fit_transform(df_norm.to_numpy())  # Scores matrix
    H_full = nmf.components_.T  # Basis matrix (preys x components)

    # Assign preys to their strongest component
    prey_component_df = assign_preys_to_primary_component(df_norm, H_full)

    # Save prey assignments
    prey_component_df.to_csv(f"{save_path}nmf_component_bait_size_preys_primary_component.csv", index=False)


    ### **Step 3: Run GSEA on Low-Correlation Components**
    def run_gsea_for_low_correlation_components(prey_component_df, low_correlation_df):
        """
        Runs GO:CC enrichment for preys in low-correlation components.

        Parameters:
        - prey_component_df: DataFrame with preys assigned to primary NMF components.
        - low_correlation_df: DataFrame of low-correlation components (< 0.50).
        - output_folder: Folder to save enrichment results.

        Returns:
        - Saves a CSV file for each low-correlation component’s GO:CC enrichment results.
        """
        gp = GProfiler(return_dataframe=True)

        # Get unique low-correlation components
        low_corr_components = low_correlation_df["Component"].unique()

        for component in low_corr_components:
            # Get preys assigned to this component
            preys = prey_component_df[prey_component_df["Primary_Component"] == component]["Prey"].tolist()

            # Run GO enrichment (only GO:CC terms)
            if preys:
                go_df = gp.profile(organism="hsapiens", query=preys)
                go_df = go_df[go_df["source"] == "GO:CC"]  # Keep only GO Cellular Component terms

                # Save results
                filename = f"{save_path}nmf_component_bait_size_GSEA_LowCorrelation_Component_{component}.csv"
                go_df.to_csv(filename, index=False)
                print(f"Saved GO:CC results for Low-Correlation Component {component} to {filename}")

    # Run GSEA for low-correlation components
    run_gsea_for_low_correlation_components(prey_component_df, low_correlation_df)


