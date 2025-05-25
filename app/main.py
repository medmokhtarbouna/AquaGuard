import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fpdf import FPDF

# === Charger modèle & scaler ===
pipeline = joblib.load("../models/xgboost_balanced_model_3.pkl")
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["xgb"]


# === Titre ===
st.set_page_config(page_title="Qualité de l'eau", layout="centered")
st.title("💧 Analyse intelligente de la qualité de l’eau")
st.markdown("Cette application prédit la **potabilité** de l’eau selon les normes OMS et recommande l’usage optimal.")

# st.markdown("---")

# === Références spécifiques par usage (source: document PDF) ===
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


normes_r = {
    "Potable": {
        "pH": "6.5–8.5", "Turbidité": "< 5", "Conductivité": "< 500",
        "TDS (Solids)": "< 500", "Sulfate": "< 250", "Chloramines": "1–4",
        "Hardness": "< 300", "Organic Carbon": "< 15", "Trihalométhanes": "< 80"
    },
    "Irrigation": {
        "pH": "6.0–8.5", "Turbidité": "< 10", "Conductivité": "< 2250",
        "TDS (Solids)": "< 2000", "Sulfate": "< 1000", "Chloramines": "Non utilisé",
        "Hardness": "Acceptable", "Organic Carbon": "Non spécifié", "Trihalométhanes": "Non utilisé"
    },
    "Industriel": {
        "pH": "6.0–9.0", "Turbidité": "< 25", "Conductivité": "< 3000",
        "TDS (Solids)": "< 1500", "Sulfate": "< 500", "Chloramines": "< 4",
        "Hardness": "< 500", "Organic Carbon": "<= 10", "Trihalométhanes": "Rare"
    },
    "Environnement": {
        "pH": "6.5–9.0", "Turbidité": "< 10", "Conductivité": "< 1000",
        "TDS (Solids)": "< 1000", "Sulfate": "< 400", "Chloramines": "< 0.1",
        "Hardness": "Non spécifié", "Organic Carbon": "<= 5", "Trihalométhanes": "< 50"
    },
}



# ⬇️ Affichage des normes dans un expander unique
with st.expander("📌 Références des normes OMS par usage"):
    usage_select = st.selectbox("Choisissez un usage de référence", list(normes_r.keys()))
    st.dataframe(pd.DataFrame([normes_r[usage_select]], index=[usage_select]).T)


# === Normes OMS ===
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

default_values = {
    "pH": 7.2,
    "Temperature": 20.0,
    "Conductivity @25°C": 450.0,
    "Dissolved Oxygen": 7.0,
    "BOD - 5 days (Total)": 2.0,
    "Ammonia-Total (as N)": 0.3,
    "ortho-Phosphate (as P) - unspecified": 0.1,
    "Total Hardness (as CaCO3)": 150.0,
    "True Colour": 10.0
}


st.subheader("🔢 Entrez les paramètres mesurés :")

# 📥 Partie Upload CSV
uploaded_file = st.file_uploader("📥 Ou importer un fichier CSV", type=["csv"])

input_values = {}

# Exemple de colonnes attendues
expected_columns = [
    "pH", "Temperature", "Conductivity @25°C", "Dissolved Oxygen",
    "BOD - 5 days (Total)", "Ammonia-Total (as N)",
    "ortho-Phosphate (as P) - unspecified", "Total Hardness (as CaCO3)", "True Colour"
]

if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        if not all(col in df_uploaded.columns for col in expected_columns):
            st.error("❌ Le fichier CSV doit contenir exactement ces colonnes :")
            st.code(", ".join(expected_columns))
            st.stop()

        # Utiliser la première ligne
        input_values = df_uploaded.iloc[0][expected_columns].to_dict()
        st.success("✅ Données importées avec succès depuis le fichier CSV.")

    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        st.stop()

else:
    # 📥 Partie saisie manuelle si aucun fichier importé
    cols = st.columns(2)
    for i, feature in enumerate(expected_columns):
        col = cols[i % 2]
        input_values[feature] = col.number_input(
            feature, min_value=0.0, max_value=1000.0, value=default_values.get(feature, 0.0)
        )



# === Prédiction
if st.button("🔬 Prédire la potabilité"):

    input_df = pd.DataFrame([input_values])
    input_df = input_df[list(normes.keys())]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Affichage résultat
    st.subheader("📈 Résultat de la prédiction :")
    if prediction == 1:
        st.success("✅ L’eau est **potable** selon le modèle d’IA.")
    else:
        st.error("❌ L’eau **n’est pas potable** selon le modèle d’IA.")

    # Vérification conformité
    dépassements, conformes = [], []
    for feat, (min_v, max_v) in normes.items():
        val = input_values[feat]
        if min_v <= val <= max_v:
            conformes.append((feat, val))
        else:
            dépassements.append((feat, val, min_v, max_v))
    pct_conformité = 100 * len(conformes) / len(normes)

    st.subheader("🧪 Comparaison aux normes OMS")
    st.markdown(f"🧾 **Taux de conformité :** `{pct_conformité:.1f}%`")

    if dépassements:
        st.warning("🚨 **Paramètres hors normes :**")
        for feat, val, min_v, max_v in dépassements:
            st.markdown(f"🔸 **{feat}** = `{val}` _(hors plage [{min_v} - {max_v}])_")
    else:
        st.success("✅ Aucun dépassement détecté.")

    # Affichage graphique
    st.subheader("📊 Comparaison graphique")
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, (feat, (min_v, max_v)) in enumerate(normes.items()):
        val = input_values[feat]
        ax.barh(i, max_v, color="#b4e2c4")
        ax.barh(i, val, color="#00d26a" if min_v <= val <= max_v else "red")
    ax.set_yticks(range(len(normes)))
    ax.set_yticklabels(normes.keys())
    ax.invert_yaxis()
    ax.set_title("Valeurs mesurées vs. Normes OMS")
    st.pyplot(fig)

    # Suggestion usage optimal
    scores = {}
    for usage, ref_values in valeurs_par_usage.items():
        match = 0
        for feature, (min_v, max_v) in normes.items():
            val = input_values[feature]
            if min_v <= val <= max_v:
                match += 1
        scores[usage] = match / len(normes) * 100
    usage = max(scores, key=scores.get)

    # Téléchargement CSV & Excel
    st.subheader("📤 Télécharger les résultats")
    result_data = input_values.copy()
    result_data["Potabilité_IA"] = "Potable" if prediction == 1 else "Non potable"
    result_data["Usage_recommandé"] = usage
    result_data["Conformité_%"] = f"{pct_conformité:.1f}"
    result_df = pd.DataFrame([result_data])

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Télécharger en CSV", csv, "résultat_eau.csv", "text/csv")

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False, sheet_name="Résultat")
    st.download_button("⬇️ Télécharger en Excel", excel_buffer.getvalue(), "résultat_eau.xlsx")


    # Définir une classe PDF
    class WaterQualityPDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 14)
            self.cell(0, 10, 'Rapport d\'analyse de la qualité de l\'eau', ln=True, align='C')
            self.ln(10)

        def add_section_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, ln=True)
            self.ln(5)

        def add_line(self, label, value):
            self.set_font('Arial', '', 11)
            self.cell(0, 8, f"{label}: {value}", ln=True)

        def add_blank_line(self, height=5):
            self.ln(height)

    # === Générer le PDF en mémoire
    pdf = WaterQualityPDF()
    pdf.add_page()

    pdf.add_section_title("Résultats principaux")
    pdf.add_line("Potabilité IA", "Potable" if prediction == 1 else "Non potable")
    pdf.add_line("Usage recommandé", best_usage if 'best_usage' in locals() else "Non défini")
    pdf.add_line("Taux de conformité", f"{pct_conformité:.1f}%")

    pdf.add_blank_line(10)

    pdf.add_section_title("Paramètres mesurés")
    for feature, value in input_values.items():
        pdf.add_line(feature, value)

    # Enregistrer dans un buffer mémoire
    pdf_buffer = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_data = pdf_buffer.getvalue()

    # === Bouton de téléchargement
    st.download_button("⬇️ Télécharger en PDF", pdf_bytes, file_name="rapport_eau.pdf", mime="application/pdf")



# === 🔍 Détection de l’usage optimal ===
if st.button("✨ Suggérer l’usage optimal de cette eau"):

    scores = {}
    for usage, ref_values in valeurs_par_usage.items():
        match = 0
        for feature, (min_v, max_v) in normes.items():
            val = input_values[feature]
            if min_v <= val <= max_v:
                match += 1
        scores[usage] = match / len(normes) * 100  # % conformité

    # Trier les scores
    best_usage = max(scores, key=scores.get)
    st.subheader("💡 Usage optimal suggéré :")
    st.success(f"✅ {best_usage} ({scores[best_usage]:.1f}% conformité aux normes OMS)")

    # Affichage des scores pour comparaison
    with st.expander("📊 Voir le taux de conformité pour chaque usage"):
        for usage, score in scores.items():
            st.markdown(f"- **{usage}** : `{score:.1f}%`")

