from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
# from database import create_engine_mysql
from datetime import date, time

# Charger le modèle entraîné
model = joblib.load('./models/best_model.pkl')

# Initialiser FastAPI
app = FastAPI()

# Ajout du Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les methodes (POST, GET, etc.)
    allow_headers=["*"],  # Autoriser tous les headers
)
# Définir les caractéristiques attendues par le modèle
class EnergyPredictionRequest(BaseModel):
    Date: date
    Heures: time
    Meteo_Irradience: float
    PR_Total_Compteurs: float
    Ensoleillement: float
    PR_Reference: float
    Nombre_Panneau: int
    Nombre_Onduleur: int

elements = [
    'Date',
    'Heures',
    'Meteo_Irradience',
    'PR_Total_Compteurs',
    'Ensoleillement',
    'PR_Reference',
    'Nombre_Panneau',
    'Nombre_Onduleur'
]

# Point d'entrée pour prédire l'énergie horaire
@app.post("/predict/")
def predict_energy(data: EnergyPredictionRequest):
    # Transformer les données d'entrée en un tableau NumPy 2D avec une seule ligne
    input_data = np.array([[getattr(data, element) for element in elements]])
    input_data = pd.DataFrame(input_data, columns=elements)
    heure = int(str(input_data['Heures'].iloc[0]).split(":")[0])
    
    # Faire une prédiction
    prediction = model.predict(input_data)
    
    # Appliquer la transformation inverse pour obtenir les valeurs originales
    predicted_energy = np.expm1(prediction[0])  # np.expm1(x) = exp(x) - 1
    
    # Retourner la prédiction
    return JSONResponse(content={"predicted_energy": predicted_energy,
                                 "Time": heure})


# Point d'entrée pour vérifier l'état du service
@app.get("/")
def read_root():
    return {"message": "Energy prediction API is up and running"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host='127.0.0.1', port=8000)