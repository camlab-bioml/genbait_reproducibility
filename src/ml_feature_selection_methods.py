import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from collections import defaultdict
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

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

    scores_df = pd.DataFrame(scores)
    basis_df = pd.DataFrame(basis)

    X = df_norm.T.to_numpy()
    y = []
    for i in range(basis.shape[0]):
        max_rank = np.argmax(basis[i,:]) 
        y.append(max_rank)
    y = np.asarray(y) 

    scores_80 = nmf.fit_transform(df_80)
    basis_80 = nmf.components_.T

    scores_80_df = pd.DataFrame(scores_80)
    basis_80_df = pd.DataFrame(basis_80)

    X = df_norm.T.to_numpy()
    y_80 = []
    for i in range(basis_80.shape[0]):
        max_rank = np.argmax(basis_80[i,:]) 
        y_80.append(max_rank)
    y_80 = np.asarray(y_80)    
    # Initialize basis_20 with zeros (its dimensions will be n x p)
    basis_20 = np.zeros((scores_80.shape[1], df_20.shape[1]))

    # Iterate over each column in df_20 and solve for each column in basis_20
    for i in range(df_20.shape[1]):
        basis_20[:, i], _ = nnls(scores_80, df_20.iloc[:, i])

    # Now, basis_20 should be the matrix you're looking for

    basis_20 = basis_20.T
    basis_20_df = pd.DataFrame(basis_20)
    df_80_20 = pd.concat([df_80, df_20], axis=1)
    basis_80_20 = pd.concat([basis_80_df, basis_20_df], axis=0)
    basis_80_20.reset_index(inplace=True)

    y_20 = []
    for i in range(basis_20.shape[0]):
        max_rank = np.argmax(basis_20[i,:]) 
        y_20.append(max_rank)
    y_20 = np.asarray(y_20)   

    X_train, X_test, y_train, y_test = df_80.T, df_20.T, y_80, y_20

    return X_train, y_train, df_80_20

def feature_selection_methods(df_norm, save_path, seed=4):
    X_train, y_train, df_80_20 = prepare_feature_selection_data(df_norm, seed)
    k = len(df_80_20)

    # Apply feature selection using Chi-squared test
    selector_chi2 = SelectKBest(chi2, k=k)
    selector_chi2.fit(X_train,y_train)
    chi2_indices = np.argsort(selector_chi2.scores_)[-k:][::-1]

    # Apply feature selection using ANOVA F-test
    selector_f_classif = SelectKBest(f_classif, k=k)
    selector_f_classif.fit(X_train,y_train)
    f_classif_indices = np.argsort(selector_f_classif.scores_)[-k:][::-1]

    # Apply feature selection using Mutual Information
    selector_mutual_info_classif = SelectKBest(score_func=lambda X, y: mutual_info_classif(X_train, y_train, random_state=46), k=k)
    selector_mutual_info_classif.fit(X_train,y_train)
    selector_mutual_classif_scores = selector_mutual_info_classif.scores_
    mutual_info_classif_indices = np.argsort(selector_mutual_classif_scores)[-k:][::-1]

    # Lasso
    lasso = LogisticRegression(penalty='l1', solver='saga', random_state=46)
    lasso.fit(X_train, y_train)
    lasso_coef = lasso.coef_
    selected_features_lasso = np.argsort(np.abs(lasso_coef).mean(axis=0))[::-1][:k]  

    # Ridge
    ridge = LogisticRegression(penalty='l2', solver='saga', random_state=46)
    ridge.fit(X_train, y_train)
    ridge_coef = ridge.coef_
    selected_features_ridge = np.argsort(np.abs(ridge_coef).mean(axis=0))[::-1][:k] 

    # Elastic Net penalty
    elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=46)
    elastic_net.fit(X_train, y_train)
    elastic_net_coef = elastic_net.coef_
    selected_features_elastic_net = np.argsort(np.abs(elastic_net_coef).mean(axis=0))[::-1][:k]  

    # Random Forest
    random_forest = RandomForestClassifier(random_state=46)
    random_forest.fit(X_train, y_train)
    rf_feature_importances = random_forest.feature_importances_
    selected_features_rf = np.argsort(rf_feature_importances)[::-1][:k]  # Select top 50 features

    # Gradient Boosting Classifier
    gbm = GradientBoostingClassifier(random_state=46)
    gbm.fit(X_train, y_train)
    gbm_feature_importances = gbm.feature_importances_
    selected_features_gbm = np.argsort(gbm_feature_importances)[::-1][:k]  # Select top 50 features

    # XGBoost Classifier
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, random_state=46, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    xgb_feature_importances = xgb_model.feature_importances_
    selected_features_xgb = np.argsort(xgb_feature_importances)[::-1][:k]  # Select top 50 features

    chi2_baits = []
    for i in chi2_indices:
        chi2_baits.append(df_80_20.index[i])

    f_classif_baits = []
    for i in f_classif_indices:
        f_classif_baits.append(df_80_20.index[i])

    mutual_info_classif_baits = []
    for i in mutual_info_classif_indices:
        mutual_info_classif_baits.append(df_80_20.index[i])    
        
    lasso_baits = []
    for i in selected_features_lasso:
        lasso_baits.append(df_80_20.index[i])

    ridge_baits = []
    for i in selected_features_ridge:
        ridge_baits.append(df_80_20.index[i])

    elastic_net_baits = []
    for i in selected_features_elastic_net:
        elastic_net_baits.append(df_80_20.index[i]) 
        
    rf_baits = []
    for i in selected_features_rf:
        rf_baits.append(df_80_20.index[i])

    gbm_baits = []
    for i in selected_features_gbm:
        gbm_baits.append(df_80_20.index[i])

    xgb_baits = []
    for i in selected_features_xgb:
        xgb_baits.append(df_80_20.index[i])


    df = pd.DataFrame({'chi_2':chi2_baits, 'f_classif':f_classif_baits, 'mutual_info_classif':mutual_info_classif_baits,
                    'lasso':lasso_baits, 'ridge':ridge_baits, 'elastic_net':elastic_net_baits,
                    'rf':rf_baits, 'gbm':gbm_baits, 'xgb':xgb_baits})
    df.to_csv(f'{save_path}/selected_baits_with_training_train-test-0.8_seed{seed}.csv')

    
    return df


def run_feature_selection_for_seeds(df_norm, save_path):
    seeds = range(10)
    result_dfs = []

    for seed in seeds:
        df_result = feature_selection_methods(df_norm, save_path, seed)
        result_dfs.append(df_result)

    return result_dfs


