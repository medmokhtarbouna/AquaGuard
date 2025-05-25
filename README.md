# 💧 AquaGuard – Analyse intelligente de la qualité de l’eau

AquaGuard est une application intelligente basée sur l'IA pour **analyser la qualité de l’eau** selon les **normes de l’OMS** et recommander l’usage optimal (potable, irrigation, domestique, industriel ou environnemental).

---

## 🚀 Fonctionnalités principales

* 🔬 Prédiction de la **potabilité de l’eau** via un modèle XGBoost entraîné.
* 📈 Vérification de la **conformité aux normes OMS** pour chaque paramètre.
* 💡 Suggestion de l’**usage optimal** basé sur les normes de qualité par contexte.
* 📅 Prise en charge de la **saisie manuelle** ou du **téléversement CSV**.
* 📄 Export des résultats en **CSV, Excel, PDF**.
* 📁 Visualisations interactives via **Streamlit**.

---

## 🧠 Modèles ML utilisés

* `RandomForestClassifier`
* `XGBoostClassifier`
* Pipelines avec **scaler**, **balancing**, et **feature engineering**
* Enregistrés dans `models/` sous forme `.pkl`

---

## 🗂️ Structure du projet

```
AquaGuard/
├── app/
│   └── main.py                  # Interface Streamlit
├── data/                        # Données sources (si utilisées)
├── models/                      # Modèles ML enregistrés
├── notebooks/                   # Jupyter Notebooks pour EDA, Modelling, etc.
├── outputs/                     # Résultats générés
├── presentation/                # Slides ou contenu de présentation
├── script/
│   └── models/                  # Scripts de prétraitement et d'entraînement
├── src/                         # Logique métier
├── utils/                       # Fonctions d’analyse et de recommandation
├── requirements.txt             # Dépendances Python
└── README.md
```

---

## 🛠️ Installation

1. Clonez le dépôt :

```bash
git clone https://github.com/votre-utilisateur/AquaGuard.git
cd AquaGuard
```

2. Créez un environnement virtuel et activez-le :

```bash
python -m venv venv
source venv/bin/activate  # Sous Windows : venv\Scripts\activate
```

3. Installez les dépendances :

```bash
pip install -r requirements.txt
```

---

## 👤 Lancement de l'application

```bash
cd app
streamlit run main.py
```
```bash
cd app
python -m streamlit run main.py
```

---

## 📄 Exemple de fichier CSV attendu

```csv
pH,Temperature,Conductivity @25°C,Dissolved Oxygen,BOD - 5 days (Total),Ammonia-Total (as N),ortho-Phosphate (as P) - unspecified,Total Hardness (as CaCO3),True Colour
7.1,23.0,450,6.8,3.2,0.3,0.1,160,10
```

---

## 📚 Références

* Normes de qualité de l’eau – Organisation Mondiale de la Santé (OMS)
* Algorithmes d’IA : scikit-learn, xgboost
* Visualisation : matplotlib, Streamlit
* Génération PDF : FPDF

---

## 🧑‍💻 Auteurs

* **Votre Nom** – Développeur ML & Data Science
* \[Lien LinkedIn, GitHub, Email selon besoin]

---

## 📜 Licence

Ce projet est sous licence MIT – voir le fichier `LICENSE` pour plus de détails.
