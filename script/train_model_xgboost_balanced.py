# script/train_model_xgboost_balanced.py

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# 📌 1. Charger les données originales et les échantillons augmentés
df_original = pd.read_csv("../data/processed/water_labeled_by_normes.csv")
df_augmented = pd.read_csv("../data/processed/augmented_potable_samples.csv")

# 📌 2. Aligner les colonnes
df_augmented["Potability"] = np.nan  # colonne présente dans df_original
df_combined = pd.concat([df_original, df_augmented], ignore_index=True)

# 📌 3. Séparer X et y
X = df_combined.drop(columns=["Potability", "Is_Potable_By_Normes"])
y = df_combined["Is_Potable_By_Normes"]

# 📌 4. Split des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 📌 5. Construction du pipeline
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("xgb", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        base_score=0.5,
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=42
    ))
])

# 📌 6. Entraînement
pipeline.fit(X_train, y_train)

# 📌 7. Évaluation
y_pred = pipeline.predict(X_test)
print("📊 Rapport de classification - XGBoost équilibré :\n")
print(classification_report(y_test, y_pred, digits=4))

# 📌 8. Sauvegarde du modèle
joblib.dump(pipeline, "../models/xgboost_balanced.pkl")
print("✅ Modèle XGBoost équilibré sauvegardé dans: models/xgboost_balanced.pkl")
