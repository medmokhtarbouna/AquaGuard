# train_model_balanced.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Charger les données
df = pd.read_csv("../data/raw/water_potability.csv")

# Supprimer les lignes avec des valeurs manquantes
df_clean = df.dropna()

# Séparation des données
X = df_clean.drop("Potability", axis=1)
y = df_clean["Potability"]

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Pipeline avec normalisation robuste
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Entraînement
pipeline.fit(X_train, y_train)

# Évaluation
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Sauvegarde du modèle
joblib.dump(pipeline, "../models/random_forest_balanced.pkl")
