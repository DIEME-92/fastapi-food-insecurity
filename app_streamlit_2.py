import streamlit as st
import requests
import matplotlib.pyplot as plt
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ✅ Chargement des données
@st.cache
def load_data_raw():
    return pd.read_csv("data.csv")   # Base AVANT normalisation
data = load_data_raw()

# fonction importation des données
@st.cache(persist=True)
def load_data():
# 📥 Chargement du dataset nettoyé avant application
#chemin_projet = "D:/PROJET_DIT-20250506T153458Z-001/MES_PROJETS/"
    df = pd.read_csv("data_encoded_1.csv")
    return df

# affichage de la table de données 
df = load_data()
df_sample =df.sample(100)
if st.sidebar.checkbox("Afficher les données brutes", False):
    st.subheader("Jeu de données 'data_encoded_1.csv' : Echantillon de 100 observateurs")
    st.write(df_sample)

seed = 123

st.title("📊 Analyse exploratoire du dataset")
st.write("Voici quelques statistiques descriptives sur les réponses des participants.")
st.subheader("Hauteur : DOUDOU DIEME")

st.subheader("📌 Statistiques descriptives")
st.dataframe(df.describe().round(2))

variables = [
    "q606_1_avoir_faim_mais_ne_pas_manger",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q603_sauter_un_repas",
    "q601_ne_pas_manger_nourriture_saine_nutritive"
]

# Matrice de corrélation des variables
st.subheader("📈 Matrice de corrélation des variables")
fig, ax = plt.subplots(figsize=(20, 10))
corr = df[variables].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("🧠 Interprétation des corrélations")

# Fonction d'interprétation des variables
def interpret_correlation(value):
    if abs(value) < 0.1:
        strength = "Corrélation négligeable"
    elif abs(value) < 0.3:
        strength = "Corrélation faible"
    elif abs(value) < 0.5:
        strength = "Corrélation modérée"
    elif abs(value) < 0.7:
        strength = "Corrélation forte"
    else:
        strength = "Corrélation très forte"

    direction = "positive" if value > 0 else "négative"
    return f"{strength} ({direction})"

# Construction du tableau d'interprétation exclut des doublons
interpretation_rows = []
seen_pairs = set()

for var1 in variables:
    for var2 in variables:
        if var1 != var2:
            pair = tuple(sorted([var1, var2]))  # ex: avec ("q601", "q603")

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                coef = corr.loc[var1, var2]
                interpretation_rows.append({
                    "Variable 1": pair[0],
                    "Variable 2": pair[1],
                    "Corrélation": round(coef, 3),
                    "Interprétation": interpret_correlation(coef)
                })

# Affichage dans l'l'application Streamlit
st.write("Voici l'interprétation automatique des corrélations entre les variables :")
st.dataframe(pd.DataFrame(interpretation_rows))
########################################
########################################
######
st.sidebar.subheader("📊 Sélection des variables à afficher")

# ✅ Option Multiselect dans la sidebar pour l'affichage des histogrammes
vars_selectionnees = st.sidebar.multiselect(
    "Choisissez les variables pour afficher leurs histogrammes :",
    variables
)

# ✅ Choix de palette de couleurs automatiques pour chaque histogramme
couleurs = sns.color_palette("husl", len(vars_selectionnees))

# ✅ Affichage en colonnes des histogrammes (2 à 2 par ligne)
if vars_selectionnees:
    cols = st.columns(2)
    index = 0

    for var, couleur in zip(vars_selectionnees, couleurs):
        with cols[index % 2]:
            st.subheader(f"Histogramme : {var}")
            fig, ax = plt.subplots()
            sns.histplot(df[var], bins=10, kde=True, color=couleur, ax=ax)
            ax.set_title(f"Distribution de : {var}")
            st.pyplot(fig)

        index += 1

##########################################################################

#selected_var = st.selectbox("📌 Choisissez une variable à explorer :", variables)

#fig, ax = plt.subplots()
#sns.histplot(df[selected_var], bins=10, kde=True, color='skyblue', ax=ax)
#ax.set_title(f"Distribution de : {selected_var}")
#st.pyplot(fig)

##############################"""PREDICTION DE L'INSECURITE ALIMENTAIRE """###################################################
##############################################################################################
###########################################   data ##################################"
# ✅ Prévalence IA par région
st.subheader("📍 Prévalence de l'insécurité alimentaire par région")

data["IA_binaire"] = data["insécurité_alimentaire"].isin(["modérée", "sévère"]).astype(int)

prevalence_region = (
    data.groupby("q100_region")["IA_binaire"]
    .mean()
    .reset_index(name="Prévalence (%)")
)

prevalence_region["Prévalence (%)"] = (prevalence_region["Prévalence (%)"] * 100).round(2)

st.dataframe(prevalence_region)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=prevalence_region, x="q100_region", y="Prévalence (%)", palette="viridis", ax=ax)
ax.set_title("Prévalence de l'insécurité alimentaire par région")
plt.xticks(rotation=45)
st.pyplot(fig)


# ✅ Prévalence IA par département
st.subheader("📍 Prévalence de l'insécurité alimentaire par département")

prevalence_dept = (
    data.groupby("q101_departement")["IA_binaire"]
    .mean()
    .reset_index(name="Prévalence (%)")
)

prevalence_dept["Prévalence (%)"] = (prevalence_dept["Prévalence (%)"] * 100).round(2)

st.dataframe(prevalence_dept)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=prevalence_dept, x="q101_departement", y="Prévalence (%)", palette="magma", ax=ax)
ax.set_title("Prévalence de l'insécurité alimentaire par département")
plt.xticks(rotation=45)
st.pyplot(fig)
################################################################################################
#################################################################################################"
###################################  data ######################""
st.set_page_config(page_title="Prédiction Insécurité Alimentaire", page_icon="🍽️")

st.title("🧠 Prédiction d'insécurité alimentaire")
st.write("Indiquez vos réponses ci-dessous pour obtenir une prédiction.")

# 🔹 Voici le formulaire utilisateur
q606 = st.number_input("Combien de fois avez-vous eu faim sans manger ?", min_value=0, max_value=10, value=0)
q605 = st.number_input("Combien de fois avez-vous manqué de nourriture par manque d'argent ?", min_value=0, max_value=10, value=0)
q604 = st.number_input("Combien de fois avez-vous mangé moins que nécessaire ?", min_value=0, max_value=10, value=0)
q603 = st.number_input("Combien de repas avez-vous sauté aujourd'hui ?", min_value=0, max_value=10, value=0)
q601 = st.number_input("Combien de fois avez-vous mangé une nourriture peu nutritive ?", min_value=0, max_value=10, value=0)

# 🔹 Affichage du bouton de prédiction
if st.button("🔍 Lancer la prédiction"):
    payload = {
        "q606_1_avoir_faim_mais_ne_pas_manger": q606,
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": q605,
        "q604_manger_moins_que_ce_que_vous_auriez_du": q604,
        "q603_sauter_un_repas": q603,
        "q601_ne_pas_manger_nourriture_saine_nutritive": q601
    }

    try:
        response = requests.post("https://fastapi-food-insecurity.onrender.com/predict", json=payload)
        result = response.json()

        niveau = result.get("niveau", "inconnu")
        score = result.get("score", 0.00)
        profil = result.get("profil", "inconnu")
        probabilites = result.get("probabilités", {})

        # 🔹 Affichage du niveau
        if niveau == "sévère":
            st.error("🔴 Niveau d'insécurité alimentaire : **sévère**")
        elif niveau == "modérée":
            st.warning("🟠 Niveau d'insécurité alimentaire : **modérée**")
        else:
            st.success("🟢 Aucun signe d'insécurité alimentaire")

        # 🔹 Barre de score
        st.write("### 🔎 Score de risque")
        st.progress(score)

        # 🔹 Profil
        st.write(f"Profil détecté : **{profil.capitalize()}**")

        # 🔹 Graphique circulaire
        st.write("### 📊 Répartition des probabilités")
        fig, ax = plt.subplots()
        labels = list(probabilites.keys())
        sizes = list(probabilites.values())
        colors = ['#FDBE85', '#FF6F61']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la requête : {e}")
