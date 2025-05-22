# train_model_balanced.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report

# Charger le fichier CSV
df = pd.read_csv("../data/raw/water_potability.csv")

# Supprimer les lignes avec des valeurs manquantes
df_clean = df.dropna()

# Séparer les variables explicatives et la cible
X = df_clean.drop("Potability", axis=1)
y = df_clean["Potability"]

# Séparer les données en entraînement/test (stratifié)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Construire le pipeline : normalisation + modèle
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ))
])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Évaluer le modèle
y_pred = pipeline.predict(X_test)
print("📊 Rapport de classification :\n")
print(classification_report(y_test, y_pred, digits=4))

# Sauvegarder le modèle
joblib.dump(pipeline, "../models/random_forest_balanced.pkl")
print("✅ Modèle sauvegardé avec succès.")
