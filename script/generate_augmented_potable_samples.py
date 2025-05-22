# script/generate_augmented_potable_samples.py

import pandas as pd
import numpy as np

# 📌 Nombre d'échantillons à générer
N = 500

# 📌 Fonctions de génération aléatoire dans les plages OMS
def random_pH(): return round(np.random.uniform(6.5, 8.5), 2)
def random_hardness(): return round(np.random.uniform(80, 280), 2)
def random_solids(): return round(np.random.uniform(100, 490), 2)
def random_chloramines(): return round(np.random.uniform(1.0, 4.0), 2)
def random_sulfate(): return round(np.random.uniform(100, 240), 2)
def random_conductivity(): return round(np.random.uniform(200, 480), 2)
def random_organic_carbon(): return round(np.random.uniform(2, 14.5), 2)
def random_trihalomethanes(): return round(np.random.uniform(10, 79), 2)
def random_turbidity(): return round(np.random.uniform(0.5, 4.9), 2)

# 📌 Génération des données
data = []
for _ in range(N):
    row = [
        random_pH(),
        random_hardness(),
        random_solids(),
        random_chloramines(),
        random_sulfate(),
        random_conductivity(),
        random_organic_carbon(),
        random_trihalomethanes(),
        random_turbidity(),
        1  # Is_Potable_By_Normes
    ]
    data.append(row)

# 📌 Création DataFrame
columns = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
    "Is_Potable_By_Normes"
]

df_augmented = pd.DataFrame(data, columns=columns)

# 📌 Enregistrement dans fichier
df_augmented.to_csv("../data/processed/augmented_potable_samples.csv", index=False)

print("✅ Échantillons augmentés enregistrés avec succès.")
