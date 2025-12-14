import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ✅ Chargement des données
@st.cache(persist=True)
def load_data():
    df = pd.read_csv("data_encoded_1.csv")
    return df

df = load_data()
df_sample = df.sample(100)

if st.sidebar.checkbox("Afficher les données brutes", False):
    st.subheader("Jeu de données 'data_encoded_1.csv' : Echantillon de 100 observateurs")
    st.write(df_sample)

st.title("📊 Analyse exploratoire du dataset")
st.subheader("📌 Statistiques descriptives")
st.dataframe(df.describe().round(2))

variables = [
    "q606_1_avoir_faim_mais_ne_pas_manger",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q603_sauter_un_repas",
    "q601_ne_pas_manger_nourriture_saine_nutritive"
]

# 🔹 Matrice de corrélation
st.subheader("📈 Matrice de corrélation des variables")
fig, ax = plt.subplots(figsize=(20, 10))
corr = df[variables].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

########################################
# 🔹 Performances des modèles avec sélecteur
########################################

# 📋 Performance - Random Forest
rf_perf = pd.DataFrame({
    "Métrique": ["Accuracy", "AUC", "Recall"],
    "Train": [0.996152, 0.986885, 0.973770],
    "Test": [0.994231, 0.981481, 0.962963]
})

# 📋 Performance - XGBoost
xgb_perf = pd.DataFrame({
    "Métrique": ["Accuracy", "AUC", "Recall"],
    "Train": [0.996152, 0.986885, 0.973770],
    "Test": [0.994231, 0.981481, 0.962963]
})

# 🔹 Sélecteur de modèle dans la sidebar
st.sidebar.subheader("⚙️ Choix du modèle à afficher")
modele = st.sidebar.selectbox("Sélectionnez un modèle :", ["Random Forest", "XGBoost"])

# 🔹 Affichage conditionnel
if modele == "Random Forest":
    st.subheader("📋 Performance - Random Forest")
    st.dataframe(rf_perf)

    fig, ax = plt.subplots()
    rf_perf.set_index("Métrique")[["Train","Test"]].plot(kind="bar", ax=ax, color=["#4CAF50","#2196F3"])
    ax.set_title("Random Forest - Performance")
    st.pyplot(fig)

elif modele == "XGBoost":
    st.subheader("📋 Performance - XGBoost")
    st.dataframe(xgb_perf)

    fig, ax = plt.subplots()
    xgb_perf.set_index("Métrique")[["Train","Test"]].plot(kind="bar", ax=ax, color=["#FF9800","#9C27B0"])
    ax.set_title("XGBoost - Performance")
    st.pyplot(fig)

########################################
# 🔹 Formulaire de prédiction
########################################
st.title("🧠 Prédiction d'insécurité alimentaire")
q606 = st.number_input("Combien de fois avez-vous eu faim sans manger ?", min_value=0, max_value=10, value=0)
q605 = st.number_input("Combien de fois avez-vous manqué de nourriture par manque d'argent ?", min_value=0, max_value=10, value=0)
q604 = st.number_input("Combien de fois avez-vous mangé moins que nécessaire ?", min_value=0, max_value=10, value=0)
q603 = st.number_input("Combien de repas avez-vous sauté aujourd'hui ?", min_value=0, max_value=10, value=0)
q601 = st.number_input("Combien de fois avez-vous mangé une nourriture peu nutritive ?", min_value=0, max_value=10, value=0)

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

        if niveau == "sévère":
            st.error("🔴 Niveau d'insécurité alimentaire : **sévère**")
        elif niveau == "modérée":
            st.warning("🟠 Niveau d'insécurité alimentaire : **modérée**")
        else:
            st.success("🟢 Aucun signe d'insécurité alimentaire")

        st.write("### 🔎 Score de risque")
        st.progress(score)

        st.write(f"Profil détecté : **{profil.capitalize()}**")

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
