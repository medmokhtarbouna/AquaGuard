# utils/analyse_exceeding.py

def analyser_non_conformité(valeurs):
    """
    Prend un dictionnaire de valeurs {indicateur: valeur} et retourne
    une liste des anomalies détectées + une phrase d'explication.
    """
    seuils = {
        "pH": (6.5, 8.5),
        "Turbidité": (0, 5),
        "Conductivité": (0, 500),
        "TDS (Solids)": (0, 500),
        "Sulfate": (0, 250),
        "Chloramines": (1, 4),
        "Hardness": (0, 300),
        "Organic Carbon": (0, 15),
        "Trihalométhanes": (0, 80),
    }

    anomalies = []

    for indicateur, valeur in valeurs.items():
        if indicateur not in seuils:
            continue

        borne_min, borne_max = seuils[indicateur]

        if valeur < borne_min:
            anomalies.append(f"🔻 {indicateur} = {valeur} (en dessous de {borne_min})")
        elif valeur > borne_max:
            anomalies.append(f"🔺 {indicateur} = {valeur} (au-dessus de {borne_max})")

    # Synthèse textuelle
    if anomalies:
        message = "❌ Certains indicateurs dépassent les normes OMS :\n\n"
        message += "\n".join(anomalies)
        message += "\n\n💡 L’eau est considérée comme non potable car elle présente un ou plusieurs dépassements critiques."
    else:
        message = "✅ Aucun dépassement détecté par rapport aux normes OMS."

    return message
