# script/train_model_xgboost.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# 🔹 1. Chargement des données labellisées par les normes OMS
df = pd.read_csv("../data/processed/water_labeled_by_normes.csv")

# 🔹 2. Définir les variables d'entrée et de sortie
X = df.drop(columns=["Potability", "Is_Potable_By_Normes"])
y = df["Is_Potable_By_Normes"]

# 🔹 3. Séparation entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 🔹 4. Pipeline avec XGBoost et RobustScaler
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("xgb",XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        base_score=0.5, 
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=42
    ))
])

# 🔹 5. Entraînement
pipeline.fit(X_train, y_train)

# 🔹 6. Évaluation du modèle
y_pred = pipeline.predict(X_test)
print("📊 Rapport XGBoost basé sur les normes OMS :\n")
print(classification_report(y_test, y_pred, digits=4))

# 🔹 7. Sauvegarde du modèle
joblib.dump(pipeline, "../models/xgboost_by_normes.pkl")
print("✅ Modèle XGBoost sauvegardé dans: models/xgboost_by_normes.pkl")
