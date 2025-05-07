import pandas as pd

# Load input files
bait_df = pd.read_csv("datasets/baits-latest.txt", sep="\t")
saint_df = pd.read_csv("datasets/saint-latest.txt", sep="\t")

# Filter SAINT for BFDR ≤ 0.01 and count preys per bait
prey_counts = saint_df[saint_df['BFDR'] <= 0.01].groupby("Bait")["Prey"].nunique().reset_index()
prey_counts.columns = ["bait", "PreyCount"]

# Merge with expected localization(s)
bait_df = bait_df[["bait", "expected localization(s)"]]
merged_df = pd.merge(prey_counts, bait_df, on="bait")

# Split multi-compartment annotations
merged_df["expected localization(s)"] = merged_df["expected localization(s)"].str.split(", ")
exploded_df = merged_df.explode("expected localization(s)").rename(columns={"expected localization(s)": "Compartment"})

# Manual A selection function
def select_manual_a(df, target_count):
    compartment_counts = df["Compartment"].value_counts()
    alloc = (compartment_counts / compartment_counts.sum()) * target_count
    alloc = alloc.round().astype(int)
    diff = target_count - alloc.sum()
    if diff != 0:
        alloc.iloc[0] += diff
    selected_rows = []
    for comp, k in alloc.items():
        top_k = df[df["Compartment"] == comp].sort_values("PreyCount", ascending=False).head(k)
        selected_rows.append(top_k)
    final = pd.concat(selected_rows).drop_duplicates(subset=["bait"])
    return final["bait"].reset_index(drop=True)

# Run selections
panel_40 = select_manual_a(exploded_df, 40)
panel_60 = select_manual_a(exploded_df, 60)
panel_80 = select_manual_a(exploded_df, 80)

# Combine into one DataFrame
final_df = pd.DataFrame({
    "ManualA_40": panel_40.reindex(range(max(len(panel_40), len(panel_60), len(panel_80)))),
    "ManualA_60": panel_60.reindex(range(max(len(panel_40), len(panel_60), len(panel_80)))),
    "ManualA_80": panel_80.reindex(range(max(len(panel_40), len(panel_60), len(panel_80))))
})

# Save to CSV
final_df.to_csv("targeted_approach_selected_baits_highest_number_of_preys_proportional.csv", index=False)
