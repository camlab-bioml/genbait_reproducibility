import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib

# Configure matplotlib for publication-ready plots
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def load_expression_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def normalize_gene_name(gene_name):
    return gene_name.split('_')[0]

def plot_baits_expression_heatmap(
    bait_file,
    uniprot_mapping_file,
    expression_dir,
    output_dir,
):
    # Define cell lines to include
    cell_lines = ["HEK-293", "HeLa", "LNCaP", "U2-OS", "MCF-7", "GaMG", "SW-620", "A-375", "A-549", "K-562", "Jurkat"]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load baits
    baits_df = pd.read_csv(bait_file)
    selected_baits = baits_df.iloc[:, 0].tolist()

    # Load UniProt mapping
    uniprot_mapping = load_expression_data(uniprot_mapping_file)

    # Normalize bait names
    normalized_baits = [normalize_gene_name(bait) for bait in selected_baits]
    bait_to_uniprot = {bait: uniprot_mapping.get(bait, None) for bait in normalized_baits}

    # Initialize expression matrix
    expression_matrix = pd.DataFrame(index=cell_lines, columns=selected_baits)

    # Fill expression matrix
    for cell in cell_lines:
        cell_path = os.path.join(expression_dir, f"{cell}_expression_normalized.pkl")
        if os.path.exists(cell_path):
            data = load_expression_data(cell_path)
            for bait in selected_baits:
                norm_bait = normalize_gene_name(bait)
                uniprot_id = bait_to_uniprot.get(norm_bait)
                value = data.get(uniprot_id, 0) if uniprot_id else 0
                expression_matrix.at[cell, bait] = value
        else:
            expression_matrix.loc[cell] = 0

    expression_matrix = expression_matrix.astype(float).fillna(0)
    expression_matrix.to_csv(os.path.join(output_dir, 'baits_expression_heatmap.csv'))

    # Plot heatmap
    plt.figure(figsize=(max(12, len(selected_baits) * 0.5), 8))
    sns.heatmap(expression_matrix, cmap="viridis", linewidths=0.5, linecolor='gray')
    plt.xlabel("Baits", fontsize=16)
    plt.ylabel("Cell Lines", fontsize=16)
    plt.title("Normalized Expression of Baits Across Cell Lines", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baits_expression_heatmap.pdf'), dpi=300)
    plt.close()

    # ---- Stats: exclude non-relevant baits ---- #
    unmapped_baits = {"CALR3", "CYP2C1", "HIST1H2BG", "SV40"}
    filtered_baits = [bait for bait in selected_baits if normalize_gene_name(bait) not in unmapped_baits]
    stats_matrix = expression_matrix[filtered_baits]

    # Binary matrix for expression presence
    binary_expr = stats_matrix > 0
    n_cell_lines = len(cell_lines)

    # Calculate percentages
    expressed_in_all = (binary_expr.sum(axis=0) == n_cell_lines).sum()
    expressed_in_half_or_more = (binary_expr.sum(axis=0) >= (n_cell_lines // 2)).sum()
    total_baits = len(stats_matrix.columns)

    print(f"Baits expressed in all cell lines: {expressed_in_all}/{total_baits} ({(expressed_in_all / total_baits) * 100:.1f}%)")
    print(f"Baits expressed in ≥50% of cell lines: {expressed_in_half_or_more}/{total_baits} ({(expressed_in_half_or_more / total_baits) * 100:.1f}%)")

    # Median and standard deviation per bait
    stats_df = stats_matrix.transpose().agg(['median', 'std'], axis=1)
    stats_df.to_csv(os.path.join(output_dir, "bait_expression_median_std_filtered.csv"))
    print(stats_df.describe())
