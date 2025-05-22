# === train_model_xgboost.py ===

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

# 1. Charger les données propres et étiquetées
df = pd.read_csv("../data/processed/ireland_clean_labeled.csv")

# 2. Séparer les caractéristiques et la cible
X = df.drop(columns=["Is_Potable_By_Normes"])
y = df["Is_Potable_By_Normes"]

# 3. Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Normalisation robuste
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entraîner le modèle XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Évaluation du modèle
y_pred = model.predict(X_test_scaled)
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 7. Enregistrement du modèle et du scaler
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/xgboost_by_normes_2.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("\n✅ Modèle et scaler sauvegardés avec succès.")
