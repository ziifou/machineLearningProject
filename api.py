from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = FastAPI(
    title="API de Machine Learning",
    description="API pour entraîner un modèle et faire des prédictions",
    version="1.0.0"
)

class TrainData(BaseModel):
    data: list
    target: list

class PredictData(BaseModel):
    data: list

@app.post("/training", tags=["Model Training"], summary="Entraîner un modèle")
def train_model(train_data: TrainData):
    try:
        print(train_data.data)
        X = pd.DataFrame(train_data.data)
        y = pd.Series(train_data.target)
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, 'model/model.pkl')
        return {"message": "Modèle entraîné avec succès"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", tags=["Model Prediction"], summary="Faire une prédiction")
def predict(data: PredictData):
    try:
        model = joblib.load('model/model.pkl')
        X = pd.DataFrame(data.data)
        predictions = model.predict(X)
        prediction =  predictions.tolist().pop()
        if prediction == 1:
            return {"prediction": "Malade"}
        else:
            return {"prediction": "Sain"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model", tags=["External API"], summary="Obtenir des informations du modèle")
def get_model_info():
    # Exemple avec OpenAI ou HuggingFace
    return {"message": "API externe appelée avec succès"}
