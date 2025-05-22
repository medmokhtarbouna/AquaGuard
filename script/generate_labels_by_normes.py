# generate_labels_by_normes.py

import pandas as pd

# 📌 Charger les données
df = pd.read_csv("../data/raw/water_potability.csv")

# 📌 Supprimer les lignes incomplètes
df_clean = df.dropna().copy()

# 📌 Définir la fonction de conformité
def est_potable(row):
    try:
        return int(
            6.5 <= row["ph"] <= 8.5 and
            row["Hardness"] < 300 and
            row["Solids"] < 500 and
            1 <= row["Chloramines"] <= 4 and
            row["Sulfate"] < 250 and
            row["Conductivity"] < 500 and
            row["Organic_carbon"] < 15 and
            row["Trihalomethanes"] < 80 and
            row["Turbidity"] < 5
        )
    except:
        return 0

# 📌 Appliquer la fonction sur chaque ligne
df_clean["Is_Potable_By_Normes"] = df_clean.apply(est_potable, axis=1)

# 📌 Enregistrer dans un nouveau fichier
df_clean.to_csv("../data/processed/water_labeled_by_normes.csv", index=False)

print("✅ Fichier généré avec succès: data/processed/water_labeled_by_normes.csv")
