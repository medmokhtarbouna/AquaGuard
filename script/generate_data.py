import pandas as pd
import numpy as np

# 🔧 Définition des normes OMS pour l'eau potable
normes = {
    "pH": (6.5, 8.5),
    "Temperature": (0, 30),
    "Conductivity @25°C": (0, 500),
    "Dissolved Oxygen": (5, 14),
    "BOD - 5 days (Total)": (0, 5),
    "Ammonia-Total (as N)": (0, 0.5),
    "ortho-Phosphate (as P) - unspecified": (0, 0.2),
    "Total Hardness (as CaCO3)": (0, 300),
    "True Colour": (0, 15)
}

# 🧠 Fonction pour générer un échantillon synthétique
def generate_sample(potable=True):
    sample = {}
    for feature, (min_v, max_v) in normes.items():
        if potable:
            val = np.random.uniform(min_v + 0.05*(max_v - min_v), max_v - 0.05*(max_v - min_v))
        else:
            if np.random.rand() < 0.5:
                val = np.random.uniform(max_v + 0.01, max_v + (max_v * 0.5) + 0.1)
            else:
                val = np.random.uniform(max(0, min_v - (min_v * 0.5) - 0.1), min_v - 0.01)
        sample[feature] = round(val, 3)
    sample["Is_Potable_By_Normes"] = int(potable)
    return sample

# 🔁 Génération des données équilibrées
n_samples = 30000
n_potable = n_samples // 2
n_non_potable = n_samples - n_potable

potable_data = [generate_sample(potable=True) for _ in range(n_potable)]
non_potable_data = [generate_sample(potable=False) for _ in range(n_non_potable)]

# 🔄 Fusionner les deux
full_data = potable_data + non_potable_data
np.random.shuffle(full_data)
df = pd.DataFrame(full_data)

# 💾 Sauvegarde du fichier
df.to_csv("../data/processed/synthetic_water_quality_balanced.csv", index=False)
