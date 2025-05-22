import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# 1. 🔹 Charger les données
df = pd.read_csv("../data/processed/synthetic_water_quality_balanced.csv")

# 2. 🔹 Séparer les features et la cible
X = df.drop(columns=["Is_Potable_By_Normes"])
y = df["Is_Potable_By_Normes"]

# 3. 🔹 Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4. 🔹 Pipeline : Scaler + XGBoost
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("xgb", XGBClassifier(
        n_estimators=300,        # réduit pour rapidité
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False, # évite l’avertissement
        eval_metric="logloss",
        random_state=42
    ))
])

# 5. 🔹 Entraînement du modèle
pipeline.fit(X_train, y_train)

# 6. 🔹 Évaluation
y_pred = pipeline.predict(X_test)
print("📊 Rapport de classification :")
print(classification_report(y_test, y_pred, digits=4))

# 7. 🔹 Sauvegarde du modèle
joblib.dump(pipeline, "../models/xgboost_balanced_model_3.pkl")
joblib.dump(pipeline.named_steps["scaler"], "../models/scaler.pkl")

print("✅ Modèle sauvegardé sous : xgboost_balanced_model_3.pkl")
