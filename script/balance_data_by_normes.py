# === balance_data_by_normes.py ===

import pandas as pd
from sklearn.utils import resample

# 📥 Charger le fichier original (données propres et étiquetées selon les normes OMS)
df = pd.read_csv("../data/processed/ireland_clean_labeled.csv")

# ✅ Séparer les classes : potable et non potable
df_potable = df[df["Is_Potable_By_Normes"] == 1]
df_non_potable = df[df["Is_Potable_By_Normes"] == 0]

# 🔁 Suréchantillonner la classe minoritaire (potable) pour équilibrer le dataset
df_potable_equilibre = resample(
    df_potable,
    replace=True,  # autoriser la duplication
    n_samples=len(df_non_potable),  # égaliser les tailles
    random_state=42
)

# 🔀 Combiner les deux classes pour former un dataset équilibré
df_equilibre = pd.concat([df_non_potable, df_potable_equilibre])

# 🎲 Réordonner aléatoirement les lignes
df_equilibre = df_equilibre.sample(frac=1, random_state=42).reset_index(drop=True)

# 💾 Enregistrer le nouveau fichier équilibré
df_equilibre.to_csv("../data/processed/ireland_water_quality_balanced.csv", index=False)

print("✅ Données équilibrées sauvegardées sous 'ireland_water_quality_balanced.csv'")
