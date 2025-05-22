# === train_model_balanced.py ===

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# 📥 Charger le dataset équilibré
df = pd.read_csv("../data/processed/ireland_water_quality_balanced.csv")

# ✅ Séparer les variables explicatives (X) et la cible (y)
X = df.drop(columns=["Is_Potable_By_Normes"])
y = df["Is_Potable_By_Normes"]

# ✂️ Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ⚙️ Pipeline avec mise à l’échelle robuste + XGBoost
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("xgb", XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ))
])

# 🚀 Entraînement du modèle
pipeline.fit(X_train, y_train)

# 📊 Évaluation
y_pred = pipeline.predict(X_test)
print("=== Matrice de confusion ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Rapport de classification ===")
print(classification_report(y_test, y_pred))

# 💾 Sauvegarde du pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "../models/xgboost_balanced_model_2.pkl")

print("\n✅ Modèle sauvegardé dans 'models/xgboost_balanced_model.pkl'")
