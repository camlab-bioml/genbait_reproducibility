
import requests
import re
import pandas as pd
import os
import pickle
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- Function to Normalize Gene Names ---
def normalize_gene_names(gene_names):
    normalized = [gene.split('_')[0] for gene in gene_names]
    return normalized

# --- Function to Map Gene Symbols to UniProt IDs ---
def map_genes_to_uniprot(gene_symbols, mapping_file):
    if os.path.exists(mapping_file):
        with open(mapping_file, "rb") as f:
            uniprot_ids = pickle.load(f)
        print(f"Loaded UniProt mapping from {mapping_file}")
    else:
        uniprot_ids = {}
        for gene_symbol in gene_symbols:
            if gene_symbol not in uniprot_ids:
                req = requests.get(f'https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_symbol}+AND+taxonomy_id:9606&format=json')
                if req.status_code == 200:
                    data = req.json()
                    if 'results' in data and data['results']:
                        uniprot_ids[gene_symbol] = data['results'][0]['primaryAccession']
                    else:
                        uniprot_ids[gene_symbol] = None
        with open(mapping_file, "wb") as f:
            pickle.dump(uniprot_ids, f)
    return uniprot_ids

# --- Function to Fetch Expression Data in Batches ---
def fetch_expression_in_batches(uniprot_ids, cell_line, tissue_id, output_file, batch_size=50):
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            return pickle.load(f)
    else:
        expression_data = {}
        batch_counter = 1
        for batch in batch_dict(uniprot_ids, batch_size):
            print(f"[{cell_line}] Processing batch {batch_counter}...")
            for gene, uniprot_id in batch.items():
                if uniprot_id not in expression_data:
                    url = f"https://www.proteomicsdb.org/proteomicsdb/logic/api/proteinexpression.xsodata/InputParams"
                    query = f"(PROTEINFILTER='{uniprot_id}',MS_LEVEL=1,TISSUE_ID_SELECTION='{tissue_id}',TISSUE_CATEGORY_SELECTION='cell line',SCOPE_SELECTION=1,GROUP_BY_TISSUE=1,CALCULATION_METHOD=0,EXP_ID=-1)/Results?$select=NORMALIZED_INTENSITY&$format=json"
                    response = requests.get(f"{url}{query}")
                    if response.status_code == 200:
                        data = response.json().get("d", {}).get("results", [])
                        expression_data[uniprot_id] = float(data[0]["NORMALIZED_INTENSITY"]) if data else None
            print(f"[{cell_line}] Completed batch {batch_counter}.")
            batch_counter += 1
            with open(output_file, "wb") as f:
                pickle.dump(expression_data, f)
        print(f"[{cell_line}] All batches completed.")
        return expression_data

# --- Helper Function for Batching ---
def batch_dict(data, batch_size):
    iterator = iter(data.items())
    for _ in range(0, len(data), batch_size):
        yield dict(islice(iterator, batch_size))

# --- Function to Process Each Cell Line ---
def process_cell_line(row, original_matrix, uniprot_mapping, cell_line_path):
    cell_line_name = row["TISSUE_NAME"].replace(" cell", "")
    tissue_id = row["TISSUE_ID"]
    bto_dict = {'HEK-293': 'BTO:0000007', cell_line_name: tissue_id}

    # Fetch expression data
    hek293_data = fetch_expression_in_batches(uniprot_mapping, "HEK-293", bto_dict["HEK-293"], f"{cell_line_path}HEK-293_expression_normalized.pkl", batch_size=50)
    current_cell_data = fetch_expression_in_batches(uniprot_mapping, cell_line_name, tissue_id, f"{cell_line_path}{cell_line_name}_expression_normalized.pkl", batch_size=50)

    # Adjust the bait-prey interaction matrix
    adjusted_matrix = original_matrix.copy()
    missing_proteins = []
    for bait in original_matrix.index:
        for prey in original_matrix.columns:
            bait_uniprot = uniprot_mapping.get(normalize_gene_names([bait])[0])
            prey_uniprot = uniprot_mapping.get(normalize_gene_names([prey])[0])
            if bait_uniprot and prey_uniprot:
                hek293_bait_exp = hek293_data.get(bait_uniprot)
                current_cell_bait_exp = current_cell_data.get(bait_uniprot)
                hek293_prey_exp = hek293_data.get(prey_uniprot)
                current_cell_prey_exp = current_cell_data.get(prey_uniprot)
                if None not in [hek293_bait_exp, current_cell_bait_exp, hek293_prey_exp, current_cell_prey_exp]:
                    adjustment_factor = (current_cell_bait_exp / hek293_bait_exp) * (current_cell_prey_exp / hek293_prey_exp)
                    adjusted_matrix.loc[bait, prey] *= adjustment_factor
                else:
                    if bait_uniprot or prey_uniprot:
                        missing_proteins.append((bait, prey))

    # Save the missing proteins and adjusted matrix
    missing_proteins_df = pd.DataFrame(missing_proteins, columns=["Bait", "Prey"])
    missing_proteins_df.to_csv(f"{cell_line_path}missing_proteins_{cell_line_name}.csv", index=False)
    output_filename = f"{cell_line_path}adjusted_prey_bait_matrix_{cell_line_name}_normalized.csv"
    adjusted_matrix.to_csv(output_filename)
    return f"Completed processing for {cell_line_name}"

def generate_simulated_expression_data(df_norm, cell_line_data_file, mapping_file_path):
    # Load the filtered tissues
    filtered_tissues = cell_line_data_file
    filtered_tissues = filtered_tissues[(filtered_tissues["TISSUE_GROUP_NAME"].notna()) & 
                                        (filtered_tissues["TISSUE_GROUP_NAME"] != "unknown") & 
                                        (filtered_tissues["TISSUE_ID"].str.startswith("BTO")) & 
                                        (filtered_tissues["TISSUE_NAME"] != "HEK-293 cell")]

    print(filtered_tissues)

    original_matrix = df_norm
    all_genes = list(set(original_matrix.index) | set(original_matrix.columns))
    normalized_genes = normalize_gene_names(all_genes)
    uniprot_mapping = map_genes_to_uniprot(normalized_genes, mapping_file_path)

    # Use ThreadPoolExecutor with max_workers=2 to limit to 2 cell lines at a time
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_cell_line, row, original_matrix, uniprot_mapping) for _, row in filtered_tissues.iterrows()]
        for future in as_completed(futures):
            print(future.result())



