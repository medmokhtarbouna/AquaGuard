# script/train_model_by_normes.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report

# 📌 Chargement du fichier nettoyé avec labels OMS
df = pd.read_csv("../data/processed/water_labeled_by_normes.csv")

# 📌 Définition des features et de la cible
X = df.drop(columns=["Potability", "Is_Potable_By_Normes"])
y = df["Is_Potable_By_Normes"]

# 📌 Division entraînement/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 📌 Construction du pipeline
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ))
])

# 📌 Entraînement
pipeline.fit(X_train, y_train)

# 📌 Évaluation
y_pred = pipeline.predict(X_test)
print("📊 Rapport de classification basé sur les normes OMS :\n")
print(classification_report(y_test, y_pred, digits=4))

# 📌 Sauvegarde du modèle
joblib.dump(pipeline, "../models/random_forest_by_normes.pkl")
print("✅ Modèle sauvegardé dans: models/random_forest_by_normes.pkl")
