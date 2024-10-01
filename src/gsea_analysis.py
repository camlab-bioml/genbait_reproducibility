# gsea_analysis.py

import pandas as pd
from gprofiler import GProfiler

def perform_gsea_analysis(basis_matrix_original, basis_matrix_subset_reordered_reduced, df_norm, df_subset_reduced, y_original, y_subset, file_path='gsea_results'):
    """
    Perform GSEA on the provided matrices and save results to Excel files.
    """
    basis_original_df = pd.DataFrame(basis_matrix_original, index=df_norm.columns)
    basis_subset_reordered_reduced_df = pd.DataFrame(basis_matrix_subset_reordered_reduced, index=df_subset_reduced.columns)
    basis_original_df['Label'] = y_original
    basis_subset_reordered_reduced_df['Label'] = y_subset

    gp = GProfiler(return_dataframe=True)

    with pd.ExcelWriter(f'{file_path}/GSEA_basis_original.xlsx') as writer:
        for name, group in basis_original_df.groupby('Label'):
            prey_names = list(group.index)
            go_df = gp.profile(organism='hsapiens', query=prey_names)
            go_df = go_df[go_df['source'] == 'GO:CC'] # type: ignore

            # Create a sanitized version of the name for the sheet name
            sheet_name = str(name).replace('/', '').replace('\\', '').replace('?', '').replace('*', '').replace('[', '').replace(']', '')[:31]
            go_df.to_excel(writer, sheet_name=sheet_name)

    with pd.ExcelWriter(f'{file_path}/GSEA_basis_subset.xlsx') as writer:
        for name, group in basis_subset_reordered_reduced_df.groupby('Label'):
            prey_names = list(group.index)
            go_df = gp.profile(organism='hsapiens', query=prey_names)
            go_df = go_df[go_df['source'] == 'GO:CC'] # type: ignore

            # Create a sanitized version of the name for the sheet name
            sheet_name = str(name).replace('/', '').replace('\\', '').replace('?', '').replace('*', '').replace('[', '').replace(']', '')[:31]
            go_df.to_excel(writer, sheet_name=sheet_name)

