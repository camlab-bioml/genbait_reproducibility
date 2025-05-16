import os
import pickle
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def plot_nmf_score_vs_expression():

    # List of cell lines
    cell_lines = [
        "HeLa", "LNCaP", "U2-OS", "MCF-7",
        "GaMG", "SW-620", "A-375", "A-549", "K-562", "Jurkat"
    ]

    # Load HEK-293 expression data
    with open("../results/dataset1/cell_lines/HEK-293_expression_normalized.pkl", "rb") as f:
        hek_expr = pickle.load(f)

    # Store results
    expression_correlations = []
    nmf_means = []

    # For each cell line (except HEK-293), compute correlation + NMF score
    for cl in cell_lines:
        if cl == "HEK-293":
            continue

        # Load expression
        with open(f"../results/dataset1/cell_lines/{cl}_expression_normalized.pkl", "rb") as f:
            expr = pickle.load(f)

        # Get overlapping UniProt IDs
        common_ids = set(hek_expr).intersection(expr)

        # Prepare aligned vectors (excluding missing values)
        hek_vals = []
        cl_vals = []
        for uid in common_ids:
            if hek_expr[uid] is not None and expr[uid] is not None:
                hek_vals.append(hek_expr[uid])
                cl_vals.append(expr[uid])

        # Compute Pearson correlation of expression
        corr, _ = pearsonr(hek_vals, cl_vals)
        expression_correlations.append((cl, corr))

        # Load NMF scores for this cell line
        with open(f"../results/dataset1/simulations/{cl}/nmf_scores_ga.pkl", "rb") as f:
            scores = pickle.load(f)

        # Compute mean score (across sizes and seeds)
        all_scores = []
        for size_dict in scores.values():
            for vals in size_dict.values():
                all_scores.extend(vals)
        mean_score = sum(all_scores) / len(all_scores)
        nmf_means.append((cl, mean_score))

    # Combine into DataFrame
    df = pd.DataFrame({
        "CellLine": [cl for cl, _ in expression_correlations],
        "ExpressionCorrWithHEK293": [corr for _, corr in expression_correlations],
        "MeanNMFScore": [score for _, score in nmf_means]
    })

    # Save or plot (optional)
    df.to_csv("../results/dataset1/plots/genbait_nmfscore_vs_correlation.csv", index=False)

    # Plot
    plt.scatter(df["ExpressionCorrWithHEK293"], df["MeanNMFScore"])

    # Label each dot
    for cl, x, y in zip(cell_lines, df["ExpressionCorrWithHEK293"], df["MeanNMFScore"]):
        plt.text(x , y - 0.0008, cl, fontsize=7, ha="center", va="center")

        import numpy as np
        from scipy.stats import pearsonr

        # Fit linear regression line
        slope, intercept = np.polyfit(df["ExpressionCorrWithHEK293"], df["MeanNMFScore"], 1)
        x_vals = np.array(df["ExpressionCorrWithHEK293"])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color='gray', label='Linear fit', linestyle='solid')

        # Calculate and display Pearson correlation
        r_value, p_value = pearsonr(df["ExpressionCorrWithHEK293"], df["MeanNMFScore"])
        plt.text(
            0.75, 0.45,
            f"Pearson r = {r_value:.2f}",
            transform=plt.gca().transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
        )
        # plt.legend()

    plt.xlabel("Expression similarity to HEK-293 (Pearson correlation)")
    plt.ylabel("Mean NMF Pearsin correlation score")
    # plt.title("GENBAIT Performance vs. Expression Similarity")
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/dataset1/plots/genbait_nmfscore_vs_correlation_labeled.pdf", dpi=300)
