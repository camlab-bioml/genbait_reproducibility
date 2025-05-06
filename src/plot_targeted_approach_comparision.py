import pandas as pd
import numpy as np
from itertools import islice
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

def baits_with_most_preys(saint_filepath, original_baits_filepath, num_baits_list, output_filepath):
    # Load the datasets
    df = pd.read_csv(saint_filepath, sep='\t')
    original_baits = pd.read_csv(original_baits_filepath)['original_baits'].to_list()

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
    sorted_bait_prey_counts = list(sorted(bait_prey_counts.keys(), key=lambda bait: bait_prey_counts[bait], reverse=True))

    # Step 4: Select the top N baits for each length in num_baits_list
    selected_baits_dict = {num_baits: list(islice(sorted_bait_prey_counts, num_baits)) for num_baits in num_baits_list}

    # Step 5: Create a DataFrame with three columns (one for each bait length)
    max_baits = max(num_baits_list)
    results_df = pd.DataFrame({num_baits: selected_baits_dict[num_baits] + [''] * (max_baits - len(selected_baits_dict[num_baits])) for num_baits in num_baits_list})

    # Save to CSV
    results_df.to_csv(output_filepath, index=False)
    print(f"Saved selected baits to {output_filepath}")



import torch.optim as optim
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.decomposition import NMF
from scipy.optimize import nnls
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import random
from deap import base, creator, tools, algorithms
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

# Cache for previously computed fitness values
fitness_cache = {}

# Pre-computed values for original data
original_data_values = {}

def precompute_original_data(df_norm, n_components):
    """
    Precompute and store NMF and other values for the original data.
    """
    original_data = df_norm.to_numpy()
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_original = nmf.fit_transform(original_data)
    basis_matrix_original = nmf.components_.T
    original_data_values['scores_matrix'] = scores_matrix_original
    original_data_values['basis_matrix'] = basis_matrix_original

def evalSubsetCorrelation(df_norm, n_components, subset_range, individual):
    """
    Evaluate the fitness of an individual subset of features.
    """
    individual_key = tuple(individual)  # Convert individual to tuple to use as dictionary key
    
    if individual_key in fitness_cache:
        return fitness_cache[individual_key]  # Retrieve previously computed fitness value

    subset_indices = [i for i in range(len(individual)) if individual[i] == 1]
    
    if not (subset_range[0] <= len(subset_indices) <= subset_range[1]):
        return 0,  # Penalize invalid individuals heavily

    # Subset the data and apply NMF
    subset_data = df_norm.to_numpy()[subset_indices, :]
    nmf = NMF(n_components=n_components, init='nndsvd', l1_ratio=1, random_state=46)
    scores_matrix_subset = nmf.fit_transform(subset_data)
    basis_matrix_subset = nmf.components_.T

    # Use precomputed values for original data
    scores_matrix_original = original_data_values['scores_matrix']
    basis_matrix_original = original_data_values['basis_matrix']

    # Calculate cosine similarity and cost matrix for Hungarian algorithm
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]
    
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:n_components, n_components:]
    
    # Extract relevant statistics from the correlation matrix
    diagonal = np.diag(corr_matrix)
    diagonal_mean = np.mean(diagonal)
    diagonal_min = np.min(diagonal)
    penalty_factor = 0.5
    num_negative_values = np.sum(diagonal < 0)
    fitness = diagonal_mean - penalty_factor * num_negative_values,

    fitness_cache[individual_key] = fitness  # Cache the computed fitness value
    return fitness


def create_initial_individual(subset_range, n_features):
    """Create an individual with a number of selected features within the subset_range."""
    num_selected_features = random.randint(subset_range[0], subset_range[1])
    selected_indices = random.sample(range(n_features), num_selected_features)
    individual = [0] * n_features
    for idx in selected_indices:
        individual[idx] = 1
    return individual

def run_genetic_algorithm(df_norm, n_components, subset_range, population_size, n_generations, cxpb, mutpb, seed=4):
    random.seed(seed)
    np.random.seed(seed)
    precompute_original_data(df_norm, n_components)

    fitness_cache.clear()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    n_features = df_norm.shape[0]

    # Lambda function to create an individual without arguments
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                     lambda: create_initial_individual(subset_range, n_features))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalSubsetCorrelation, df_norm, n_components, subset_range)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof

# Prepare data for feature selection
def prepare_feature_selection_data(df_norm, seed=4):
    np.random.seed(seed)

    # Splitting the data
    num_cols = df_norm.shape[1]
    num_split = int(num_cols * 0.8)
    cols = df_norm.columns.tolist()
    np.random.shuffle(cols)
    cols_80 = cols[:num_split]
    cols_20 = cols[num_split:]

    df_80 = df_norm[cols_80]
    df_20 = df_norm[cols_20]

    # NMF Decomposition for entire df_norm
    nmf = NMF(n_components=20, init='nndsvd', l1_ratio=1, random_state=46)
    scores = nmf.fit_transform(df_norm)
    basis = nmf.components_.T

    y = np.argmax(basis, axis=1)

    scores_80 = nmf.fit_transform(df_80)
    basis_80 = nmf.components_.T
    y_80 = np.argmax(basis_80, axis=1)

    # Compute basis_20 using nnls
    basis_20 = np.zeros((scores_80.shape[1], df_20.shape[1]))
    for i in range(df_20.shape[1]):
        basis_20[:, i], _ = nnls(scores_80, df_20.iloc[:, i])

    basis_20 = basis_20.T
    y_20 = np.argmax(basis_20, axis=1)

    X_train, y_train = df_80.T.to_numpy(), y_80
    X_test, y_test = df_20.T.to_numpy(), y_20

    return X_train, y_train, df_80


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LightningNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        super(LightningNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = nn.CrossEntropyLoss()(outputs, y_batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_lightning_nn(X_train, y_train, input_size, hidden_size, output_size, epochs=10, batch_size=32, learning_rate=0.001):
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LightningNN(input_size, hidden_size, output_size, learning_rate)
    trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False, logger=False)
    trainer.fit(model, dataloader)

    return model

def get_feature_importance_shap_lightning(model, X_train):
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()

    model.eval()

    def model_forward(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    explainer = shap.Explainer(model_forward, X_train)
    shap_values = explainer(X_train)

    feature_importances = np.abs(shap_values.values).mean(axis=0).sum(axis=1)

    return feature_importances










def feature_selection_methods(df_norm, seed=4):
    bait_lengths = [40, 60, 80]
    all_selected_baits = {}

    print("\n### Running feature selection ###")

    for k in bait_lengths:
        print(f"Running feature selection for bait length: {k}")
        X_train, y_train, df_80_20 = prepare_feature_selection_data(df_norm, seed)

        # Chi2
        selector_chi2 = SelectKBest(chi2, k=k)
        selector_chi2.fit(X_train, y_train)
        chi2_baits = df_80_20.index[np.argsort(selector_chi2.scores_)[-k:][::-1]]

        # F_classif
        selector_f_classif = SelectKBest(f_classif, k=k)
        selector_f_classif.fit(X_train, y_train)
        f_classif_baits = df_80_20.index[np.argsort(selector_f_classif.scores_)[-k:][::-1]]

        # Mutual Information
        selector_mutual_info = SelectKBest(mutual_info_classif, k=k)
        selector_mutual_info.fit(X_train, y_train)
        mutual_info_baits = df_80_20.index[np.argsort(selector_mutual_info.scores_)[-k:][::-1]]

        # Lasso
        lasso = LogisticRegression(penalty='l1', solver='saga', random_state=46)
        lasso.fit(X_train, y_train)
        lasso_baits = df_80_20.index[np.argsort(np.abs(lasso.coef_).mean(axis=0))[::-1][:k]]

        # Ridge
        ridge = LogisticRegression(penalty='l2', solver='saga', random_state=46)
        ridge.fit(X_train, y_train)
        ridge_baits = df_80_20.index[np.argsort(np.abs(ridge.coef_).mean(axis=0))[::-1][:k]]

        # Elastic Net
        elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=46)
        elastic_net.fit(X_train, y_train)
        elastic_net_baits = df_80_20.index[np.argsort(np.abs(elastic_net.coef_).mean(axis=0))[::-1][:k]]

        # Random Forest
        random_forest = RandomForestClassifier(random_state=46)
        random_forest.fit(X_train, y_train)
        rf_baits = df_80_20.index[np.argsort(random_forest.feature_importances_)[::-1][:k]]

        # Gradient Boosting
        gbm = GradientBoostingClassifier(random_state=46)
        gbm.fit(X_train, y_train)
        gbm_baits = df_80_20.index[np.argsort(gbm.feature_importances_)[::-1][:k]]

        # XGBoost
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, random_state=46, eval_metric='mlogloss')
        xgb_model.fit(X_train, y_train)
        xgb_baits = df_80_20.index[np.argsort(xgb_model.feature_importances_)[::-1][:k]]

        # Neural Network with SHAP
        input_size = X_train.shape[1]
        hidden_size = 64
        output_size = len(np.unique(y_train))
        model = train_lightning_nn(X_train, y_train, input_size, hidden_size, output_size)
        nn_feature_importances = get_feature_importance_shap_lightning(model, X_train)
        nn_baits = df_80_20.index[np.argsort(nn_feature_importances)[::-1][:k]]

        # GENBAIT
        n_components = 20
        subset_range = (k-10, k)
        population_size = 500
        n_generations = 1000
        cxpb = 0.3
        mutpb = 0.1
        pop, _, hof = run_genetic_algorithm(df_norm, n_components, subset_range, population_size, n_generations, cxpb, mutpb, seed)
        best_genbait_individual = hof[0]
        genbait_selected_indices = [i for i, val in enumerate(best_genbait_individual) if val == 1]
        genbait_baits = df_80_20.index[genbait_selected_indices]

        # Store selected baits
        all_selected_baits[f'Chi2_{k}'] = chi2_baits
        all_selected_baits[f'F_classif_{k}'] = f_classif_baits
        all_selected_baits[f'Mutual_Info_{k}'] = mutual_info_baits
        all_selected_baits[f'Lasso_{k}'] = lasso_baits
        all_selected_baits[f'Ridge_{k}'] = ridge_baits
        all_selected_baits[f'Elastic_Net_{k}'] = elastic_net_baits
        all_selected_baits[f'Random_Forest_{k}'] = rf_baits
        all_selected_baits[f'Gradient_Boosting_{k}'] = gbm_baits
        all_selected_baits[f'XGBoost_{k}'] = xgb_baits
        all_selected_baits[f'Neural_Network_{k}'] = nn_baits
        all_selected_baits[f'GENBAIT_{k}'] = genbait_baits

    # Save selected baits as CSV
    df_selected_baits = pd.DataFrame.from_dict(all_selected_baits, orient='index').transpose()
    df_selected_baits.to_csv("targeted_approach_selected_baits_all_methods.csv", index=False)

    print("Feature selection completed. Results saved.")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity




# Function to compute NMF mean Pearson correlation
def compute_nmf_correlation(selected_baits, df_norm, number_of_components=20):
    original_data = df_norm.to_numpy()
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    
    if len(subset_indices) > number_of_components:
        subset_data = original_data[subset_indices, :]

        nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
        scores_matrix_original = nmf.fit_transform(original_data)
        basis_matrix_original = nmf.components_.T

        scores_matrix_subset = nmf.fit_transform(subset_data)
        basis_matrix_subset = nmf.components_.T

        cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
        cost_matrix = 1 - cosine_similarity
        _, col_ind = linear_sum_assignment(cost_matrix)
        basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

        # Calculating correlation matrix
        corr_matrix = np.corrcoef(basis_matrix_original, basis_matrix_subset_reordered, rowvar=False)[:number_of_components, number_of_components:]
        return np.min(np.diag(corr_matrix))

    return np.nan  # Return NaN if there are not enough selected baits


def compute_nmf_cosine_similarity(selected_baits, df_norm, number_of_components=20):
    original_data = df_norm.to_numpy()
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    
    if len(subset_indices) > number_of_components:
        subset_data = original_data[subset_indices, :]

        nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
        scores_matrix_original = nmf.fit_transform(original_data)
        basis_matrix_original = nmf.components_.T

        scores_matrix_subset = nmf.fit_transform(subset_data)
        basis_matrix_subset = nmf.components_.T

        cosine_similarity1 = np.dot(basis_matrix_original.T, basis_matrix_subset)
        cost_matrix = 1 - cosine_similarity1
        _, col_ind = linear_sum_assignment(cost_matrix)
        basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

        cos_sim_matrix = cosine_similarity(basis_matrix_original.T, basis_matrix_subset_reordered.T)
        return np.min(np.diag(cos_sim_matrix))  # Mean diagonal values of cosine similarity matrix

    return np.nan  # Return NaN if there are not enough selected baits

import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

def compute_nmf_kl_divergence(selected_baits, df_norm, number_of_components=20):
    original_data = df_norm.to_numpy()
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    
    if len(subset_indices) > number_of_components:
        subset_data = original_data[subset_indices, :]

        nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
        scores_matrix_original = nmf.fit_transform(original_data)
        basis_matrix_original = nmf.components_.T

        scores_matrix_subset = nmf.fit_transform(subset_data)
        basis_matrix_subset = nmf.components_.T

        cosine_similarity1 = np.dot(basis_matrix_original.T, basis_matrix_subset)
        cost_matrix = 1 - cosine_similarity1
        _, col_ind = linear_sum_assignment(cost_matrix)
        basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

        # Correct normalization along columns to ensure probability distribution
        epsilon = 1e-10  # Small constant to avoid division by zero or log(0)
        basis_matrix_original_normalized = (basis_matrix_original + epsilon) / np.sum(basis_matrix_original + epsilon, axis=0)
        basis_matrix_subset_reordered_normalized = (basis_matrix_subset_reordered + epsilon) / np.sum(basis_matrix_subset_reordered + epsilon, axis=0)

        def kl_divergence(P, Q):
            """Compute the Kullback-Leibler divergence between P and Q."""
            return np.sum(np.where(P != 0, P * np.log(P / Q), 0), axis=0)

        # Compute KL divergence between each pair of components
        kl_div_matrix = np.zeros((number_of_components, number_of_components))
        for i in range(number_of_components):
            for j in range(number_of_components):
                kl_div_matrix[i, j] = kl_divergence(basis_matrix_original_normalized[:, i], basis_matrix_subset_reordered_normalized[:, j])

        return np.max(np.diag(kl_div_matrix))  # Mean diagonal values of KL divergence matrix

    return np.nan  # Return NaN if there are not enough selected baits


from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd

def compute_nmf_ari(selected_baits, df_norm, number_of_components=20, random_state=46):
    """
    Compute the Adjusted Rand Index (ARI) after performing NMF clustering 
    on the original dataset and a subset of selected baits.

    Parameters:
    selected_baits (list): List of selected bait names.
    df_norm (pd.DataFrame): The full dataset (preys x baits).
    number_of_components (int): Number of components for NMF.
    random_state (int): Random state for reproducibility.

    Returns:
    float: The ARI score comparing the clustering between original and subset.
    """

    # Step 1: Clean selected baits list
    selected_baits = [bait.strip() for bait in selected_baits if bait in df_norm.index]

    if not selected_baits:
        return np.nan  # Return NaN if no valid baits

    # Step 2: Extract subset data
    df_subset = df_norm.loc[selected_baits]

    # Step 3: Masking irrelevant data
    mask = (df_subset != 0).any(axis=0)
    df_subset_reduced = df_subset.loc[:, mask]
    mask_removed = (df_subset == 0).all(axis=0)
    df_subset_removed = df_subset.loc[:, mask_removed]

    # Step 4: NMF Clustering
    nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=random_state)

    # Fit NMF on the original dataset
    scores_matrix_original = nmf.fit_transform(df_norm)
    basis_matrix_original = nmf.components_.T

    # Fit NMF on the subset dataset
    scores_matrix_subset = nmf.fit_transform(df_subset)
    basis_matrix_subset = nmf.components_.T

    # Step 5: Assign cluster labels
    y_original = np.argmax(basis_matrix_original, axis=1)
    y_subset = np.argmax(basis_matrix_subset, axis=1)

    # Step 6: Align basis matrices using Hungarian algorithm
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    _, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Step 7: Align labels based on the subset
    reduced_cols = df_subset_reduced.columns
    basis_matrix_subset_reordered_df = pd.DataFrame(
        basis_matrix_subset_reordered, 
        columns=[i for i in range(basis_matrix_subset_reordered.shape[1])]
    )

    # Get indices of reduced columns
    reduced_indices = [df_subset.columns.get_loc(c) for c in reduced_cols]
    basis_matrix_subset_reordered_reduced = basis_matrix_subset_reordered_df.iloc[reduced_indices, :].to_numpy()

    # Step 8: Assign final cluster labels
    y_original = np.array([np.argmax(basis_matrix_original[i, :]) for i in range(basis_matrix_original.shape[0])])
    y_subset = np.array([np.argmax(basis_matrix_subset_reordered_reduced[i, :]) for i in range(basis_matrix_subset_reordered_reduced.shape[0])])

    # Step 9: Convert results into DataFrames
    basis_original_df = pd.DataFrame(basis_matrix_original, index=df_norm.columns)
    basis_subset_reordered_reduced_df = pd.DataFrame(basis_matrix_subset_reordered_reduced, index=df_subset_reduced.columns)
    basis_original_df['Label'] = y_original
    basis_subset_reordered_reduced_df['Label'] = y_subset

    # Step 10: Identify common indices
    common_indices = basis_original_df.index.intersection(basis_subset_reordered_reduced_df.index)

    # Step 11: Extract labels for common indices
    labels_original = basis_original_df.loc[common_indices, 'Label']
    labels_subset = basis_subset_reordered_reduced_df.loc[common_indices, 'Label']

    # Step 12: Compute Adjusted Rand Index (ARI)
    ari_score = adjusted_rand_score(labels_original, labels_subset)

    return ari_score



import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

def compute_component_conservation(selected_baits, df_norm, number_of_components=20):
    original_data = df_norm.to_numpy()
    subset_indices = list(df_norm.index.get_indexer(selected_baits))
    
    if len(subset_indices) > number_of_components:
        subset_data = original_data[subset_indices, :]

        # Step 1: NMF Decomposition
        nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=46)
        scores_matrix_original = nmf.fit_transform(original_data)
        basis_matrix_original = nmf.components_.T

        scores_matrix_subset = nmf.fit_transform(subset_data)
        basis_matrix_subset = nmf.components_.T

        # Step 2: Align Components using Hungarian Matching
        cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
        cost_matrix = 1 - cosine_similarity
        _, col_ind = linear_sum_assignment(cost_matrix)
        basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

        # Step 3: Compute Component Conservation
        component_preservations = []
        for component_idx in range(number_of_components):
            # Identify preys assigned to this component in the original data
            original_preys = np.where(np.argmax(basis_matrix_original, axis=1) == component_idx)[0]

            if len(original_preys) == 0:  # No preys assigned to this component in the original
                component_preservations.append(1.0)  # Full preservation if no assignments
                continue

            # Check how many of those preys are still assigned to the same component in the subset
            subset_preys = np.where(np.argmax(basis_matrix_subset_reordered, axis=1) == component_idx)[0]
            preserved_preys = len(set(original_preys).intersection(subset_preys))

            # Calculate preservation as a fraction
            preservation = preserved_preys / len(original_preys)
            component_preservations.append(preservation)

        # Step 4: Return the minimum preservation value across components
        return min(component_preservations)

    return np.nan  # Return NaN if there are not enough selected baits


from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
from gprofiler import GProfiler
from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_nmf_go_jaccard(selected_baits, df_norm, number_of_components=20, random_state=46):
    """
    Compute the mean Jaccard index between GO:CC terms of NMF components from 
    the original dataset and the subset selected by a bait selection method.

    Parameters:
    selected_baits (list): List of selected baits.
    df_norm (pd.DataFrame): The full dataset (baits x preys).
    number_of_components (int): Number of components for NMF.
    random_state (int): Random state for reproducibility.

    Returns:
    float: The mean Jaccard index comparing GO:CC terms of original vs. subset.
    """

    # Ensure all selected baits exist in df_norm
    valid_baits = [bait for bait in selected_baits if bait in df_norm.index]
    if len(valid_baits) == 0:
        print(f"Warning: No valid baits found in df_norm for selection: {selected_baits}")
        return 0  # Return 0 if no valid baits exist

    # Extract subset based on valid baits
    df_subset = df_norm.loc[valid_baits, :]

    # Step 1: Masking irrelevant data
    mask = (df_subset != 0).any(axis=0)
    df_subset_reduced = df_subset.loc[:, mask]
    mask_removed = (df_subset == 0).all(axis=0)
    df_subset_removed = df_subset.loc[:, mask_removed]

    # Step 2: NMF Clustering
    nmf = NMF(n_components=number_of_components, init='nndsvd', l1_ratio=1, random_state=random_state)

    # Fit NMF on the original dataset
    scores_matrix_original = nmf.fit_transform(df_norm)
    basis_matrix_original = nmf.components_.T

    # Fit NMF on the subset dataset
    scores_matrix_subset = nmf.fit_transform(df_subset)
    basis_matrix_subset = nmf.components_.T

    # Step 3: Align basis matrices using Hungarian algorithm
    cosine_similarity = np.dot(basis_matrix_original.T, basis_matrix_subset)
    cost_matrix = 1 - cosine_similarity
    _, col_ind = linear_sum_assignment(cost_matrix)
    basis_matrix_subset_reordered = basis_matrix_subset[:, col_ind]

    # Step 4: Get column names of df_subset_reduced
    reduced_cols = df_subset_reduced.columns

    # Convert basis_matrix_subset_reordered into DataFrame
    basis_matrix_subset_reordered_df = pd.DataFrame(
        basis_matrix_subset_reordered,
        columns=[i for i in range(basis_matrix_subset_reordered.shape[1])]
    )

    # Step 5: Get indices of reduced columns
    reduced_indices = [df_subset.columns.get_loc(c) for c in reduced_cols]

    # Filter the basis_matrix
    basis_matrix_subset_reordered_reduced = basis_matrix_subset_reordered_df.iloc[reduced_indices, :]
    basis_matrix_subset_reordered_reduced = basis_matrix_subset_reordered_reduced.to_numpy()

    # Step 6: Assign labels
    y_original = []
    for i in range(basis_matrix_original.shape[0]):
        max_rank = np.argmax(basis_matrix_original[i, :])
        y_original.append(max_rank)
    y_original = np.asarray(y_original)

    y_subset = []
    for i in range(basis_matrix_subset_reordered_reduced.shape[0]):
        max_rank = np.argmax(basis_matrix_subset_reordered_reduced[i, :])
        y_subset.append(max_rank)
    y_subset = np.asarray(y_subset)

    # Step 7: Create DataFrames with labels
    basis_original_df = pd.DataFrame(basis_matrix_original, index=df_norm.columns)
    basis_subset_reordered_reduced_df = pd.DataFrame(basis_matrix_subset_reordered_reduced, index=df_subset_reduced.columns)
    basis_original_df['Label'] = y_original
    basis_subset_reordered_reduced_df['Label'] = y_subset

    # Step 8: Initialize GProfiler
    gp = GProfiler(return_dataframe=True)

    def calculate_jaccard_index(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0  # Handle division by zero

    def go_analysis_single_query(prey_names):
        try:
            go_df = gp.profile(organism='hsapiens', query=prey_names)
            go_df = go_df[go_df['source'] == 'GO:CC']
            return set(go_df['native'])
        except Exception as e:
            print(f"Error processing GO analysis for {prey_names}: {e}")
            return set()

    def process_go_analysis_parallel(grouped_df):
        top_native_terms = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_label = {executor.submit(go_analysis_single_query, list(group.index)): label for label, group in grouped_df}
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                try:
                    top_native_terms[label] = future.result()
                except Exception as exc:
                    print(f'GO analysis generated an exception for label {label}: {exc}')
        return top_native_terms

    # Step 9: Perform GO analysis and get top terms
    top_native_original = process_go_analysis_parallel(basis_original_df.groupby('Label'))
    top_native_subset = process_go_analysis_parallel(basis_subset_reordered_reduced_df.groupby('Label'))

    # Step 10: Compute Jaccard index for each component
    jaccard_indices = []
    for label in top_native_original:
        set1 = top_native_original[label]
        set2 = top_native_subset.get(label, set())  # Use .get to avoid key errors
        jaccard_index = calculate_jaccard_index(set1, set2)
        jaccard_indices.append(jaccard_index)

    # Step 11: Compute mean Jaccard index
    mean_jaccard_index = np.min(jaccard_indices) if jaccard_indices else 0

    return mean_jaccard_index



def compute_remaining_preys(selected_baits, df_norm):
    """
    Compute the fraction of preys that remain in the dataset after selecting a subset of baits.

    Parameters:
    selected_baits (list): List of selected baits.
    df_norm (pd.DataFrame): The full dataset (baits x preys).

    Returns:
    float: The fraction of remaining preys in the subset relative to the full dataset.
    """

    # Ensure all selected baits exist in df_norm
    valid_baits = [bait for bait in selected_baits if bait in df_norm.index]
    if len(valid_baits) == 0:
        print(f"Warning: No valid baits found in df_norm for selection: {selected_baits}")
        return 0  # Return 0 if no valid baits exist

    # Extract subset based on valid baits
    df_subset = df_norm.loc[valid_baits, :]

    # Convert to numpy array for computation efficiency
    original_data = df_norm.to_numpy()
    subset_indices = list(df_norm.index.get_indexer(valid_baits))
    subset_data = original_data[subset_indices, :]

    # Count remaining preys (non-zero values across all selected baits)
    prey_count = np.count_nonzero(np.count_nonzero(subset_data, axis=0))

    # Compute fraction of remaining preys
    remaining_fraction = prey_count / df_norm.shape[1]

    return remaining_fraction

def load_and_process_gaf(file_path):
    # Load the GAF file into a DataFrame
    df = pd.read_csv(file_path, sep='\t', comment='!', header=None, dtype=str)
    
    # Set column names based on the GAF 2.1 specification
    column_names = [
        "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
        "DB_Reference", "Evidence_Code", "With_or_From", "Aspect",
        "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type",
        "Taxon", "Date", "Assigned_By", "Annotation_Extension",
        "Gene_Product_Form_ID"
    ]
    df.columns = column_names[:len(df.columns)]  # Handles cases where some optional columns might be missing
    
    # Calculate the term size for each GO term
    term_size = df.groupby('GO_ID')['DB_Object_Symbol'].nunique()
    term_size = term_size.reset_index()
    term_size.columns = ['GO_ID', 'Term_Size']
    
    # Get associated genes for each GO term
    associated_genes = df.groupby('GO_ID')['DB_Object_Symbol'].unique()
    associated_genes = associated_genes.reset_index()
    associated_genes.columns = ['GO_ID', 'Associated_Genes']
    
    # Merge the dataframes on 'GO_ID'
    merged_df = pd.merge(term_size, associated_genes, on='GO_ID')
    
    return df, merged_df


# def get_go_cc_for_genes(df, genes, merged_df, max_term_size=10000):
#     df_norm = pd.read_csv('datasets/df_norm.csv', index_col=0)
#     df_subset = df_norm.loc[genes]
#     mask = (df_subset != 0).any(axis=0)
#     df_subset_reduced = df_subset.loc[:, mask]
#     subset_df = df[df['DB_Object_Symbol'].isin(df_subset_reduced.columns)]
#     cc_df = subset_df[subset_df['Aspect'] == 'C']
#     unique_go_cc_terms = cc_df['GO_ID'].unique()
#     large_terms = merged_df[merged_df['Term_Size'] <= max_term_size]['GO_ID'].tolist()
#     filtered_terms = [term for term in unique_go_cc_terms if term in large_terms]
#     return filtered_terms

def get_go_cc_for_genes(df, genes, merged_df, max_term_size=10000):
    # Replace specific gene names
    replacement_map = {
        '10-Sep': 'SEPT10',
        '09-Sep': 'SEPT9',
    }
    
    # Replace entries in the genes list
    genes = [replacement_map.get(gene, gene) for gene in genes]

    # Load normalized dataset
    df_norm = pd.read_csv('datasets/df_norm.csv', index_col=0)

    # Subset df_norm with provided genes
    try:
        df_subset = df_norm.loc[genes]
    except KeyError as e:
        raise ValueError(f"One or more genes not found in df_norm: {e}")

    # Filter columns where any value is non-zero
    mask = (df_subset != 0).any(axis=0)
    df_subset_reduced = df_subset.loc[:, mask]

    # Subset GO terms
    subset_df = df[df['DB_Object_Symbol'].isin(df_subset_reduced.columns)]
    cc_df = subset_df[subset_df['Aspect'] == 'C']

    # Filter GO terms based on size
    unique_go_cc_terms = cc_df['GO_ID'].unique()
    large_terms = merged_df[merged_df['Term_Size'] <= max_term_size]['GO_ID'].tolist()
    filtered_terms = [term for term in unique_go_cc_terms if term in large_terms]

    return filtered_terms



def compute_go_retrieval(selected_baits, df_norm, gaf_file='datasets/goa_human.gaf', max_term_size=10000):
    """
    Compute the percentage of GO:CC terms retrieved from a subset of baits compared to the full dataset.

    Parameters:
    selected_baits (list): List of selected baits.
    df_norm (pd.DataFrame): The full dataset (baits x preys).
    gaf_file (str): Path to the GO annotation file (GAF format).
    max_term_size (int): Maximum size of GO terms to consider.

    Returns:
    float: Percentage of GO terms retrieved by the subset compared to the original baits.
    """

    # Load GAF file and process GO term associations
    df, merged_df = load_and_process_gaf(gaf_file)

    # Load the primary baits (original reference list)
    primary_baits = pd.read_csv('datasets/original_baits.csv', header=0).iloc[:, 0].to_list()

    # Compute GO:CC terms for original baits
    go_terms_original = get_go_cc_for_genes(df, primary_baits, merged_df, max_term_size)

    # Ensure all selected baits exist in df_norm
    valid_baits = [bait for bait in selected_baits if bait in df_norm.index]
    if len(valid_baits) == 0:
        print(f"Warning: No valid baits found in df_norm for selection: {selected_baits}")
        return 0  # Return 0 if no valid baits exist

    # Compute GO:CC terms for the selected baits
    go_terms_subset = get_go_cc_for_genes(df, valid_baits, merged_df, max_term_size)

    # Calculate overlap percentage between subset and original GO terms
    overlap_percentage = len(set(go_terms_subset) & set(go_terms_original)) / len(go_terms_original) * 100

    return overlap_percentage





import igraph as ig
import leidenalg

def create_knn_graph(data, k=20):
    knn_graph = kneighbors_graph(data, k, mode='connectivity', include_self=False).toarray() # type: ignore
    sources, targets = knn_graph.nonzero()
    weights = knn_graph[sources, targets]
    g = ig.Graph(directed=False)
    g.add_vertices(data.shape[0])
    edges = list(zip(sources, targets))
    g.add_edges(edges)
    g.es['weight'] = weights
    return g

def leiden_clustering(graph, resolution):
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, 
                                         weights='weight', resolution_parameter=resolution)
    return partition.membership


def precalculate_original_clusters(df_original, resolutions):
    # Transpose the original dataframe to have samples as rows
    df_original_transposed = df_original.transpose()
    
    # Dictionary to store original clusters for each resolution
    original_clusters_resolutions = {}
    
    # Perform clustering on transposed dataframe for each resolution
    for resolution in resolutions:
        original_graph = create_knn_graph(df_original_transposed, k=20)
        original_clusters = leiden_clustering(original_graph, resolution)
        original_clusters_resolutions[resolution] = original_clusters
    
    return original_clusters_resolutions

from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg
from sklearn.metrics import adjusted_rand_score

def compute_knn_leiden_ari(selected_baits, df_norm, resolutions=[0.5, 1, 1.5], k=20):
    """
    Compute Adjusted Rand Index (ARI) between original and subset clustering using k-NN & Leiden.

    Parameters:
    selected_baits (list): List of selected baits.
    df_norm (pd.DataFrame): Full dataset (baits x preys).
    resolutions (list): List of Leiden clustering resolution values.
    k (int): Number of nearest neighbors for k-NN graph construction.

    Returns:
    list: Aggregated ARI scores across all resolutions.
    """

    # Ensure selected baits exist in df_norm
    valid_baits = [bait for bait in selected_baits if bait in df_norm.index]
    if len(valid_baits) == 0:
        print(f"Warning: No valid baits found in df_norm for selection: {selected_baits}")
        return []  # Return empty list if no valid baits exist

    # Select subset of data and filter preys with nonzero values
    df_subset = df_norm.loc[valid_baits, :]
    mask = (df_subset != 0).any(axis=0)
    df_subset = df_subset.loc[:, mask]

    # Transpose for clustering (samples as rows)
    df_norm_transposed = df_norm.transpose()
    df_subset_transposed = df_subset.transpose()

    # Compute k-NN graph for original dataset
    original_graph = create_knn_graph(df_norm_transposed, k)

    # Compute Leiden clusters for original dataset at each resolution
    original_clusters_resolutions = {
        resolution: leiden_clustering(original_graph, resolution)
        for resolution in resolutions
    }

    # Compute k-NN graph for subset
    subset_graph = create_knn_graph(df_subset_transposed, k)

    # Aggregate ARI scores across all resolutions
    aggregated_ari_scores = []
    for resolution in resolutions:
        subset_clusters = leiden_clustering(subset_graph, resolution)

        # Identify common samples
        common_samples = df_norm.columns.intersection(df_subset.columns)

        # Extract cluster labels for common samples
        original_clusters = original_clusters_resolutions[resolution]
        original_index_map = {sample: idx for idx, sample in enumerate(df_norm.columns)}
        subset_index_map = {sample: idx for idx, sample in enumerate(df_subset.columns)}

        common_original_clusters = [original_clusters[original_index_map[sample]] for sample in common_samples]
        common_subset_clusters = [subset_clusters[subset_index_map[sample]] for sample in common_samples]

        # Compute ARI for this resolution and append to list
        ari_score = adjusted_rand_score(common_original_clusters, common_subset_clusters)
        aggregated_ari_scores.append(ari_score)

    return aggregated_ari_scores



from sklearn.mixture import GaussianMixture

def get_gmm_hard_assignments(data, n_clusters, seed):
    """
    Perform Gaussian Mixture Model (GMM) clustering and return hard assignments.

    Parameters:
    data (numpy array): Input data (samples x features).
    n_clusters (int): Number of clusters.
    seed (int): Random seed for reproducibility.

    Returns:
    np.array: Cluster assignments for each sample.
    """
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    gmm.fit(data)
    return gmm.predict(data)

def compute_gmm_ari(selected_baits, df_norm, cluster_numbers=[15, 20, 25, 30], seed=4):
    """
    Compute Adjusted Rand Index (ARI) between original and subset clustering using GMM.

    Parameters:
    selected_baits (list): List of selected baits.
    df_norm (pd.DataFrame): Full dataset (baits x preys).
    cluster_numbers (list): List of cluster numbers for GMM.
    seed (int): Random seed for reproducibility.

    Returns:
    list: Aggregated ARI scores across all cluster numbers.
    """

    # Ensure selected baits exist in df_norm
    valid_baits = [bait for bait in selected_baits if bait in df_norm.index]
    if len(valid_baits) == 0:
        print(f"Warning: No valid baits found in df_norm for selection: {selected_baits}")
        return []  # Return empty list if no valid baits exist

    # Select subset of data and filter preys with nonzero values
    df_subset = df_norm.loc[valid_baits, :]
    mask = (df_subset != 0).any(axis=0)
    df_subset = df_subset.loc[:, mask]

    # Identify common preys
    common_preys = df_norm.columns.intersection(df_subset.columns)

    # Transpose for clustering (samples as rows)
    df_norm_transposed = df_norm[common_preys].transpose()
    df_subset_transposed = df_subset[common_preys].transpose()

    # Precompute GMM assignments for the original dataset on common preys only
    original_assignments = {
        n_clusters: get_gmm_hard_assignments(df_norm_transposed.values, n_clusters, seed)
        for n_clusters in cluster_numbers
    }

    # Aggregate ARI scores across all cluster numbers
    aggregated_ari_scores = []
    for n_clusters in cluster_numbers:
        subset_assignments = get_gmm_hard_assignments(df_subset_transposed.values, n_clusters, seed)

        # Extract cluster labels for common samples
        original_index_map = {sample: idx for idx, sample in enumerate(df_norm[common_preys].columns)}
        subset_index_map = {sample: idx for idx, sample in enumerate(df_subset[common_preys].columns)}

        common_original_clusters = [original_assignments[n_clusters][original_index_map[sample]] for sample in common_preys]
        common_subset_clusters = [subset_assignments[subset_index_map[sample]] for sample in common_preys]

        # Compute ARI for this cluster number and append to list
        ari_score = adjusted_rand_score(common_original_clusters, common_subset_clusters)
        aggregated_ari_scores.append(ari_score)

    return aggregated_ari_scores



from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

def get_gmm_soft_assignments(data, n_clusters, seed):
    """
    Perform Gaussian Mixture Model (GMM) clustering and return soft assignments (probabilities).

    Parameters:
    data (numpy array): Input data (samples x features).
    n_clusters (int): Number of clusters.
    seed (int): Random seed for reproducibility.

    Returns:
    np.array: Cluster probability assignments for each sample.
    """
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    gmm.fit(data)
    return gmm.predict_proba(data)

def average_cluster_correlation(original_probs, subset_probs):
    """
    Compute the mean correlation between corresponding clusters in the original and subset GMM assignments.

    Parameters:
    original_probs (numpy array): Soft cluster assignments for the original dataset.
    subset_probs (numpy array): Soft cluster assignments for the subset dataset.

    Returns:
    float: Mean diagonal correlation of reordered cluster probability distributions.
    """
    corr_matrix = np.dot(original_probs.T, subset_probs)
    cost_matrix = 1 - corr_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    subset_probs_reordered = subset_probs[:, col_ind]
    reordered_corr_matrix = np.corrcoef(original_probs.T, subset_probs_reordered.T)[:original_probs.shape[1], original_probs.shape[1]:]
    diag = np.diag(reordered_corr_matrix)
    return np.mean(diag)

def compute_gmm_correlation(selected_baits, df_norm, cluster_numbers=[15, 20, 25, 30], seed=4):
    """
    Compute cluster correlation between original and subset clustering using GMM soft assignments.

    Parameters:
    selected_baits (list): List of selected baits.
    df_norm (pd.DataFrame): Full dataset (baits x preys).
    cluster_numbers (list): List of cluster numbers for GMM.
    seed (int): Random seed for reproducibility.

    Returns:
    list: Aggregated mean cluster correlation scores across all cluster numbers.
    """

    # Ensure selected baits exist in df_norm
    valid_baits = [bait for bait in selected_baits if bait in df_norm.index]
    if len(valid_baits) == 0:
        print(f"Warning: No valid baits found in df_norm for selection: {selected_baits}")
        return []  # Return empty list if no valid baits exist

    # Select subset of data and filter preys with nonzero values
    df_subset = df_norm.loc[valid_baits, :]
    mask = (df_subset != 0).any(axis=0)
    df_subset = df_subset.loc[:, mask]

    # Identify common preys
    common_preys = df_norm.columns.intersection(df_subset.columns)

    # Transpose for clustering (samples as rows)
    df_norm_transposed = df_norm[common_preys].transpose()
    df_subset_transposed = df_subset[common_preys].transpose()

    # Precompute GMM soft assignments for the original dataset on common preys only
    original_probs_dict = {
        n_clusters: get_gmm_soft_assignments(df_norm_transposed.values, n_clusters, seed)
        for n_clusters in cluster_numbers
    }

    # Aggregate correlation scores across all cluster numbers
    aggregated_correlation_scores = []
    for n_clusters in cluster_numbers:
        subset_probs = get_gmm_soft_assignments(df_subset_transposed.values, n_clusters, seed)

        # Compute mean correlation for this cluster number
        mean_corr = average_cluster_correlation(original_probs_dict[n_clusters], subset_probs)
        aggregated_correlation_scores.append(mean_corr)

    return aggregated_correlation_scores


df_norm = pd.read_csv("datasets/df_norm.csv", index_col=0)  # Normalized 
saint_filepath = "datasets/saint-latest.txt"
original_baits_filepath = "datasets/original_baits.csv"
output_filepath = "targeted_approach_selected_baits_highest_number_of_preys.csv"
bait_lengths = [40, 60, 80]

# baits_with_most_preys(saint_filepath, original_baits_filepath, bait_lengths, output_filepath)

# feature_selection_methods(df_norm)


# Load the datasets
targeted_methods = pd.read_csv("targeted_approach_selected_baits_all_methods.csv")
highest_preys = pd.read_csv("targeted_approach_selected_baits_highest_number_of_preys.csv")
highest_preys_proportional = pd.read_csv("targeted_approach_selected_baits_highest_number_of_preys_proportional.csv")

number_of_components = 20
function_to_use = compute_nmf_correlation


# Dictionary to store NMF correlation scores for each method
nmf_scores = {}

methods = ['Chi2','F_classif','Mutual_Info','Lasso','Ridge','Elastic_Net','Random_Forest','Gradient_Boosting','XGBoost','Neural_Network','GENBAIT']
# Compute for all developed methods
for method in methods:  # Get unique method names
    method_scores = []
    for length in bait_lengths:
        col_name = f"{method}_{length}"  # Correctly refer to each length column
        if col_name in targeted_methods.columns:
            selected_baits = targeted_methods[col_name].dropna().tolist()
            score = function_to_use(selected_baits, df_norm)
            method_scores.append(score)
    nmf_scores[method] = method_scores  # Store all three scores for box plot

# Compute for highest number of preys
highest_scores = []
for length in bait_lengths:
    selected_baits = highest_preys[str(length)].dropna().tolist()
    score = function_to_use(selected_baits, df_norm)
    highest_scores.append(score)
nmf_scores["High-yield baits"] = highest_scores


# Compute for highest number of preys
highest_scores_proportional = []
for length in bait_lengths:
    selected_baits = highest_preys_proportional[str(length)].dropna().tolist()
    score = function_to_use(selected_baits, df_norm)
    highest_scores_proportional.append(score)
nmf_scores["High-yield baits proportional"] = highest_scores_proportional

# Compute for literature-based markers
literature_scores = []
for length in bait_lengths:
    marker_df = pd.read_csv(f"targeted_approach_selected_baits_marker{length}.csv")
    selected_baits = marker_df["Bait"].dropna().tolist()
    score = function_to_use(selected_baits, df_norm)
    
    literature_scores.append(score)
nmf_scores["Literature Baits"] = literature_scores

# Convert to DataFrame for plotting
df_nmf_scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in nmf_scores.items()]))  # Ensure equal lengths
df_nmf_scores = df_nmf_scores.apply(lambda x: x.explode()).reset_index(drop=True)
df_nmf_scores = df_nmf_scores.apply(pd.to_numeric, errors='coerce')

print(df_nmf_scores)
# Box Plot
plt.figure(figsize=(12, 6))

df_nmf_scores.boxplot(grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
                      whiskerprops=dict(color='black'), capprops=dict(color='black'),
                      medianprops=dict(color='black'), flierprops=dict(marker='o', color='black', markersize=5, alpha=0.6))


plt.xlabel("Bait Selection Method")
plt.ylabel("NMF min KL Divergence")
# plt.title("Comparison of Bait Selection Methods Using NMF Mean Pearson Correlation")
plt.xticks(rotation=90)
plt.tight_layout()

# Save and show plot
plt.savefig("plots/tageted_approach_comparison_nmf_p.pdf", dpi=300)
plt.show()

print("Plot saved as nmf_correlation_comparison_fixed.png")