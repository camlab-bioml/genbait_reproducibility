import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def plot_baits_expression_level_comparison(bait_file, hek293_pickle, cell_pickle, uniprot_mapping_pickle, output_file, cell_line):
    """
    Loads bait selection and expression data, then plots bait expression levels in HEK293 and a target cell line.
    
    Args:
        bait_file (str): Path to the file containing selected baits.
        hek293_pickle (str): Path to HEK293 expression data pickle file.
        cell_pickle (str): Path to the target cell line expression data pickle file.
        uniprot_mapping_pickle (str): Path to the UniProt mapping pickle file.
        output_file (str): Path to save the plot.
        cell_line (str): Name of the target cell line.
    """
    # Load expression data
    def load_expression_data(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    uniprot_mapping = load_expression_data(uniprot_mapping_pickle)
    hek293_data = load_expression_data(hek293_pickle)
    cell_data = load_expression_data(cell_pickle)

    # Load selected baits
    baits_df = pd.read_csv(bait_file)
    selected_baits = baits_df.iloc[:, 0].tolist()

    # Normalize gene names and collect expression data
    def normalize_gene_name(gene_name):
        return gene_name.split('_')[0]

    expression_data = []
    for bait in selected_baits:
        normalized_name = normalize_gene_name(bait)
        uniprot_id = uniprot_mapping.get(normalized_name)
        hek293_exp = hek293_data.get(uniprot_id, 0)  # Default to 0 if missing
        cell_exp = cell_data.get(uniprot_id, 0)  # Default to 0 if missing
        expression_data.append({"Bait": bait, "Cell Line": "HEK293", "Expression Level": hek293_exp})
        expression_data.append({"Bait": bait, "Cell Line": cell_line, "Expression Level": cell_exp})

    # Convert to DataFrame
    expression_df = pd.DataFrame(expression_data)
    expression_df['Expression Level'] = expression_df['Expression Level'].astype(float)

    # Plot expression levels
    plt.figure(figsize=(12, 8))
    sns.barplot(data=expression_df, x="Bait", y="Expression Level", hue="Cell Line", dodge=True)
    plt.xticks(rotation=90)
    plt.xlabel("Baits")
    plt.ylabel("Expression Level")
    plt.title(f"Expression Levels of Selected Baits in HEK293 and {cell_line}")
    plt.legend(title="Cell Line")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

