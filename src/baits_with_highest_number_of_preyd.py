import pandas as pd
import numpy as np
from itertools import islice

df = pd.read_csv('datasets/saint-latest.txt', sep='\t')

original_baits = pd.read_csv('datasets/original_baits.csv')['original_baits'].to_list()

# Step 1: Filter the dataset for BFDR <= 0.01
filtered_df = df[df['BFDR'] <= 0.01]

# Step 2: Initialize an empty dictionary to store the counts
bait_prey_counts = {}

# Iterate over unique values in the 'Bait' column
for bait in filtered_df['Bait'].unique():
    if bait in original_baits:
        # Filter the DataFrame based on the 'Bait' value
        bait_df = filtered_df[filtered_df['Bait'] == bait]
        # Count unique 'PreyGene' values and store in the dictionary
        bait_prey_counts[bait] = bait_df['PreyGene'].nunique()

# Step 3: Sort the dictionary by values (number of PreyGenes) in descending order
sorted_bait_prey_counts = dict(sorted(bait_prey_counts.items(), key=lambda item: item[1], reverse=True))

# Now, you can do whatever you need with sorted_bait_prey_counts
print(sorted_bait_prey_counts)

first_60_items = dict(islice(sorted_bait_prey_counts.items(), 60))

print(first_60_items.keys())