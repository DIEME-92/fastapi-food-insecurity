import streamlit as st
import requests
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# fonction importation des données
@st.cache(persist=True)
def load_data():
# 📥 Chargement du dataset nettoyé
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

# Matrice de corrélation
st.subheader("📈 Matrice de corrélation des variables")
fig, ax = plt.subplots(figsize=(20, 10))
corr = df[variables].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("🧠 Interprétation des corrélations")

# Fonction d'interprétation
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

# Construction du tableau d'interprétation SANS DOUBLONS
interpretation_rows = []
seen_pairs = set()

for var1 in variables:
    for var2 in variables:
        if var1 != var2:
            pair = tuple(sorted([var1, var2]))  # ex: ("q601", "q603")

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                coef = corr.loc[var1, var2]
                interpretation_rows.append({
                    "Variable 1": pair[0],
                    "Variable 2": pair[1],
                    "Corrélation": round(coef, 3),
                    "Interprétation": interpret_correlation(coef)
                })

# Affichage dans Streamlit
st.write("Voici l'interprétation automatique des corrélations entre les variables :")
st.dataframe(pd.DataFrame(interpretation_rows))

# Calcul de la matrice
correlation_matrix = df[variables].corr()

# Seuil de corrélation forte
seuil = 0.6
fortes_corrélations = []

# Parcours des paires de variables
for i in range(len(variables)):
    for j in range(i + 1, len(variables)):
        var1 = variables[i]
        var2 = variables[j]
        corr = correlation_matrix.loc[var1, var2]
        if abs(corr) >= seuil:
            relation = "positive" if corr > 0 else "négative"
            fortes_corrélations.append(f"- **{var1}** et **{var2}** sont fortement corrélées ({relation}, coefficient = {corr:.2f})")

# Affichage
if fortes_corrélations:
    for ligne in fortes_corrélations:
        st.markdown(ligne)
    st.info("Ces corrélations suggèrent que certains comportements alimentaires sont liés entre eux. Par exemple, sauter un repas est souvent associé à manger moins que nécessaire.")
else:
    st.write("Aucune corrélation forte détectée entre les variables sélectionnées.")


st.subheader("📊 Histogrammes des variables clés")


for var in variables:
    fig, ax = plt.subplots()
    sns.histplot(df[var], bins=10, kde=True, color='orange', ax=ax)
    ax.set_title(f"Distribution de : {var}")
    st.pyplot(fig)



selected_var = st.selectbox("📌 Choisissez une variable à explorer :", variables)

fig, ax = plt.subplots()
sns.histplot(df[selected_var], bins=10, kde=True, color='skyblue', ax=ax)
ax.set_title(f"Distribution de : {selected_var}")
st.pyplot(fig)


st.set_page_config(page_title="Prédiction Insécurité Alimentaire", page_icon="🍽️")

st.title("🧠 Prédiction d'insécurité alimentaire")
st.write("Indiquez vos réponses ci-dessous pour obtenir une prédiction.")

# 🔹 Formulaire utilisateur
q606 = st.number_input("Combien de fois avez-vous eu faim sans manger ?", min_value=0, max_value=10, value=0)
q605 = st.number_input("Combien de fois avez-vous manqué de nourriture par manque d'argent ?", min_value=0, max_value=10, value=0)
q604 = st.number_input("Combien de fois avez-vous mangé moins que nécessaire ?", min_value=0, max_value=10, value=0)
q603 = st.number_input("Combien de repas avez-vous sauté aujourd'hui ?", min_value=0, max_value=10, value=0)
q601 = st.number_input("Combien de fois avez-vous mangé une nourriture peu nutritive ?", min_value=0, max_value=10, value=0)

# 🔹 Bouton de prédiction
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
