valeurs_par_usage = {
    "Eau potable": {
        "pH": (6.5, 8.5),
        "Temperature": (0, 30),
        "Conductivity @25°C": (0, 500),
        "Dissolved Oxygen": (5, 14),
        "BOD - 5 days (Total)": (0, 5),
        "Ammonia-Total (as N)": (0, 0.5),
        "ortho-Phosphate (as P) - unspecified": (0, 0.2),
        "Total Hardness (as CaCO3)": (0, 300),
        "True Colour": (0, 15)
    },
    "Irrigation": {
        "pH": (6.0, 8.5),
        "Temperature": (0, 35),
        "Conductivity @25°C": (0, 750),
        "Dissolved Oxygen": (4, 14),
        "BOD - 5 days (Total)": (0, 10),
        "Ammonia-Total (as N)": (0, 1),
        "ortho-Phosphate (as P) - unspecified": (0, 0.4),
        "Total Hardness (as CaCO3)": (0, 400),
        "True Colour": (0, 25)
    },
    "Usage domestique (hors boisson)": {
        "pH": (6.5, 9.0),
        "Temperature": (0, 40),
        "Conductivity @25°C": (0, 700),
        "Dissolved Oxygen": (3, 14),
        "BOD - 5 days (Total)": (0, 7),
        "Ammonia-Total (as N)": (0, 0.8),
        "ortho-Phosphate (as P) - unspecified": (0, 0.3),
        "Total Hardness (as CaCO3)": (0, 350),
        "True Colour": (0, 20)
    },
    "Usage industriel": {
        "pH": (6.0, 9.0),
        "Temperature": (0, 45),
        "Conductivity @25°C": (0, 1500),
        "Dissolved Oxygen": (2, 14),
        "BOD - 5 days (Total)": (0, 20),
        "Ammonia-Total (as N)": (0, 2.0),
        "ortho-Phosphate (as P) - unspecified": (0, 1.0),
        "Total Hardness (as CaCO3)": (0, 500),
        "True Colour": (0, 50)
    },
    "Milieux naturels": {
        "pH": (6.5, 8.5),
        "Temperature": (0, 25),
        "Conductivity @25°C": (0, 500),
        "Dissolved Oxygen": (6, 14),
        "BOD - 5 days (Total)": (0, 3),
        "Ammonia-Total (as N)": (0, 0.1),
        "ortho-Phosphate (as P) - unspecified": (0, 0.05),
        "Total Hardness (as CaCO3)": (0, 200),
        "True Colour": (0, 10)
    }
}

def recommander_usage_optimal(valeurs_mesurees: dict) -> str:
    scores = {}
    for usage, ref in valeurs_par_usage.items():
        score = 0
        for param, (min_val, max_val) in ref.items():
            valeur = valeurs_mesurees.get(param)
            if valeur is not None and min_val <= valeur <= max_val:
                score += 1
        taux = score / len(ref)
        scores[usage] = taux

    meilleur_usage = max(scores, key=scores.get)
    return meilleur_usage
