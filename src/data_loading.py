import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

directories = ["plots", "GA_results", "gsea_results", "random_baits"]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(filepath, sep=None, index_col=None):
    """Loads a CSV file and returns the resulting DataFrame."""
    return pd.read_csv(filepath, sep=sep, index_col=index_col)

def preprocess_data(df, primary_baits=None, file_path='data/'):
    """Preprocesses the data according to specific steps."""




    ctrls = [i.split('|') for i in df['ctrlCounts']]
    sums = [sum([int(element) for element in ctrl])/len(ctrls[0]) for ctrl in ctrls]
    df['AvgCtrl'] = sums
    df['CorrectedAvgSpec'] = df['AvgSpec'] - df['AvgCtrl']
    df = df[df['BFDR'] <= 0.01]
    df = df.pivot_table(index=['Bait'], columns=['PreyGene'], values=['CorrectedAvgSpec'])
    df = df.fillna(0)
    df = df.clip(lower=0)
    df.columns = df.columns.droplevel()
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    if primary_baits:
        df_norm = df_norm.loc[primary_baits]
        df_norm = df_norm.loc[:, (df_norm != 0).any(axis=0)]

    df_norm.to_csv(f'{file_path}df_norm.csv')
    return df_norm

