import time
import os
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
import matplotlib

# Configure matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
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



def feature_selection_methods_with_runtime(df_norm, save_path, seed=4, num_repeats=3):
    bait_lengths = [30, 60, 90]
    runtime_results_all = []
    
    for repeat in range(1, num_repeats + 1):
        print(f"\n### Running feature selection - Repeat {repeat}/{num_repeats} ###")
        all_selected_baits = {}
        runtime_results = []

        def record_runtime(method_name, bait_length, start, end):
            runtime_results.append({'method': method_name, 'bait_length': bait_length, 'runtime_seconds': end - start})

        for k in bait_lengths:
            print(f"Running feature selection for bait length: {k} (Repeat {repeat})")
            X_train, y_train, df_80_20 = prepare_feature_selection_data(df_norm, seed + repeat)

            # Chi2
            start_time = time.time()
            selector_chi2 = SelectKBest(chi2, k=k)
            selector_chi2.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('Chi2', k, start_time, end_time)
            chi2_baits = df_80_20.index[np.argsort(selector_chi2.scores_)[-k:][::-1]]

            # F_classif
            start_time = time.time()
            selector_f_classif = SelectKBest(f_classif, k=k)
            selector_f_classif.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('F_classif', k, start_time, end_time)
            f_classif_baits = df_80_20.index[np.argsort(selector_f_classif.scores_)[-k:][::-1]]

            # Mutual Information
            start_time = time.time()
            selector_mutual_info = SelectKBest(mutual_info_classif, k=k)
            selector_mutual_info.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('Mutual_Information', k, start_time, end_time)
            mutual_info_baits = df_80_20.index[np.argsort(selector_mutual_info.scores_)[-k:][::-1]]

            # Lasso
            start_time = time.time()
            lasso = LogisticRegression(penalty='l1', solver='saga', random_state=46)
            lasso.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('Lasso', k, start_time, end_time)
            lasso_baits = df_80_20.index[np.argsort(np.abs(lasso.coef_).mean(axis=0))[::-1][:k]]

            # Ridge
            start_time = time.time()
            ridge = LogisticRegression(penalty='l2', solver='saga', random_state=46)
            ridge.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('Ridge', k, start_time, end_time)
            ridge_baits = df_80_20.index[np.argsort(np.abs(ridge.coef_).mean(axis=0))[::-1][:k]]

            # Elastic Net
            start_time = time.time()
            elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=46)
            elastic_net.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('Elastic_Net', k, start_time, end_time)
            elastic_net_baits = df_80_20.index[np.argsort(np.abs(elastic_net.coef_).mean(axis=0))[::-1][:k]]

            # Random Forest
            start_time = time.time()
            random_forest = RandomForestClassifier(random_state=46)
            random_forest.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('Random_Forest', k, start_time, end_time)
            rf_baits = df_80_20.index[np.argsort(random_forest.feature_importances_)[::-1][:k]]

            # Gradient Boosting
            start_time = time.time()
            gbm = GradientBoostingClassifier(random_state=46)
            gbm.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('Gradient_Boosting', k, start_time, end_time)
            gbm_baits = df_80_20.index[np.argsort(gbm.feature_importances_)[::-1][:k]]

            # XGBoost
            start_time = time.time()
            xgb_model = xgb.XGBClassifier(use_label_encoder=False, random_state=46, eval_metric='mlogloss')
            xgb_model.fit(X_train, y_train)
            end_time = time.time()
            record_runtime('XGBoost', k, start_time, end_time)
            xgb_baits = df_80_20.index[np.argsort(xgb_model.feature_importances_)[::-1][:k]]

            # Neural Network with SHAP
            start_time = time.time()
            input_size = X_train.shape[1]
            hidden_size = 64
            output_size = len(np.unique(y_train))
            model = train_lightning_nn(X_train, y_train, input_size, hidden_size, output_size)
            nn_feature_importances = get_feature_importance_shap_lightning(model, X_train)
            end_time = time.time()
            record_runtime('Neural_Network', k, start_time, end_time)
            nn_baits = df_80_20.index[np.argsort(nn_feature_importances)[::-1][:k]]

            # GENBAIT
            start_time = time.time()
            n_components = 20
            subset_range = (k-10, k)
            population_size = 500
            n_generations = 1000
            cxpb = 0.3
            mutpb = 0.1
            pop, _, hof = run_genetic_algorithm(df_norm, n_components, subset_range, population_size, n_generations, cxpb, mutpb, seed + repeat)
            end_time = time.time()
            record_runtime('GENBAIT', k, start_time, end_time)

            # Extract selected baits from the best individual in the Hall of Fame
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

        # Save selected baits for this repeat
        df_selected_baits = pd.DataFrame.from_dict(all_selected_baits, orient='index').transpose()
        df_selected_baits.to_csv(f"{save_path}/selected_baits_repeat_{repeat}.csv", index=False)

        # Store runtime results for this repeat
        runtime_results_all.extend(runtime_results)

    # Save raw runtime results
    df_runtime_raw = pd.DataFrame(runtime_results_all)
    df_runtime_raw.to_csv(f"{save_path}/runtime_analysis_repeats.csv", index=False)

    # Compute average runtime per method and bait length
    df_runtime_avg = df_runtime_raw.groupby(['method', 'bait_length']).agg({'runtime_seconds': 'mean'}).reset_index()
    df_runtime_avg.to_csv(f"{save_path}/runtime_analysis_avg.csv", index=False)

    print("Feature selection completed for all repeats. Results saved.")

    return df_runtime_avg



def plot_runtime_analysis(df_norm, save_path):

    runtime_file = f"{save_path}/runtime_analysis_repeats.csv"


    # Check if runtime_analysis_repeats.csv exists
    if os.path.exists(runtime_file):
        print(f"Runtime file {runtime_file} found. Skipping feature selection and only plotting results.")
        df_runtime_raw = pd.read_csv(runtime_file)
    else:
        print(f"Runtime file {runtime_file} not found. Running feature selection methods.")
        df_runtime_avg = feature_selection_methods_with_runtime(df_norm, save_path, seed=4)

        # Reload the generated runtime file
        df_runtime_raw = pd.read_csv(runtime_file)

    # Compute average runtime and standard deviation
    df_runtime_avg = df_runtime_raw.groupby(['method', 'bait_length']).agg({'runtime_seconds': ['mean', 'std']}).reset_index()
    df_runtime_avg.columns = ['method', 'bait_length', 'mean_runtime', 'std_runtime']

    # Plot results
    plt.figure(figsize=(10, 6))
    methods = df_runtime_avg['method'].unique()

    for method in methods:
        df_subset = df_runtime_avg[df_runtime_avg['method'] == method]
        plt.plot(df_subset['bait_length'], df_subset['mean_runtime'], marker='o', linestyle='-', label=method)
        plt.fill_between(df_subset['bait_length'], 
                        df_subset['mean_runtime'] - df_subset['std_runtime'], 
                        df_subset['mean_runtime'] + df_subset['std_runtime'], 
                        alpha=0.2)  # Confidence interval (shaded region)

    plt.xlabel("Bait Length")
    plt.ylabel("Runtime (Seconds)")
    plt.title("Feature Selection Runtime Across Methods")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()

    plot_file = f"{save_path}/runtime_plot.pdf"
    plt.savefig(plot_file, dpi=300)
    # plt.show()

    print(f"Plot saved at {plot_file}. Runtime analysis completed.")
