# === Fichier: src/preprocess.py ===

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """
    Charger les données à partir d'un fichier CSV.
    """
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Traiter les valeurs manquantes par médiane.
    """
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

def split_features_labels(df, target_column='Potability'):
    """
    Séparer les features (X) et la cible (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def prepare_data(filepath, target_column='Potability', test_size=0.2, random_state=42):
    """
    Fonction principale : charger, nettoyer et diviser les données.
    """
    df = load_data(filepath)
    df_clean = clean_data(df)
    X, y = split_features_labels(df_clean, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
