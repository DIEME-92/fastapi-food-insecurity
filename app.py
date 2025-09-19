from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

# ✅ Initialisation FastAPI
app = FastAPI()

# ✅ Chargement du modèle ML
model = joblib.load("modele_food_insecurity.pkl")

# ✅ Variables utilisées
selected_features = [
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q606_1_avoir_faim_mais_ne_pas_manger"
]

# ✅ Schéma d'entrée
class InputData(BaseModel):
    q606_1_avoir_faim_mais_ne_pas_manger: int
    q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent: int
    q604_manger_moins_que_ce_que_vous_auriez_du: int
    q603_sauter_un_repas: int
    q601_ne_pas_manger_nourriture_saine_nutritive: int

# ✅ Endpoint de santé
@app.get("/health")
def health_check():
    return {"status": "API opérationnelle ✅"}

# ✅ Endpoint de prédiction
@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.dict()])
        input_filtered = input_df[selected_features]

        # 🔍 Cas neutre
        if input_filtered.sum().sum() == 0:
            niveau = "aucune"
            prediction_binaire = 0
            profil = "neutre"
            proba = [1.0, 0.0]
        else:
            proba = model.predict_proba(input_filtered)[0]
            seuil_severe = 0.4
            prediction_binaire = int(proba[1] > seuil_severe)
            niveau = "sévère" if prediction_binaire == 1 else "modérée"
            profil = "critique" if prediction_binaire == 1 else "intermédiaire"

        # ✅ Réponse API
        return JSONResponse(content={
            "prediction": prediction_binaire,
            "niveau": niveau,
            "profil": profil,
            "score": round(float(proba[1]), 4),
            "probabilités": {
                "classe_0": round(float(proba[0]), 4),
                "classe_1": round(float(proba[1]), 4)
            }
        }, media_type="application/json; charset=utf-8")

    except Exception as e:
        print("❌ Erreur dans /predict :", str(e))
        return JSONResponse(content={
            "error": "Une erreur est survenue lors de la prédiction.",
            "details": str(e)
        }, status_code=500, media_type="application/json; charset=utf-8")
