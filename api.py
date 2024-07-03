from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

client = OpenAI()


class Message(BaseModel):
    text: str

class Response(BaseModel):
    reply: str



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

@app.post("/training", tags=["Model Training"], summary="Entraîner un modèle", description="Entraîne un modèle de régression logistique avec les données fournies.")
def train_model(train_data: TrainData):
    """
    Entraîne un modèle de régression logistique avec les données fournies.

    - **data**: Liste des caractéristiques pour l'entraînement.
    - **target**: Liste des étiquettes cibles.
    """
    try:
        print(train_data.data)
        X = pd.DataFrame(train_data.data)
        y = pd.Series(train_data.target)
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, './model.pkl')
        return {"message": "Modèle entraîné avec succès"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", tags=["Model Prediction"], summary="Faire une prédiction", description="Prédit la classe (Malade ou Sain) pour les données fournies.")
def predict(data: PredictData):
    """
    Prédit la classe (Malade ou Sain) pour les données fournies.

    - **data**: Liste des caractéristiques pour la prédiction.
    """
    try:
        model = joblib.load('./model.pkl')
        X = pd.DataFrame(data.data)
        predictions = model.predict(X)
        prediction =  predictions.tolist().pop()
        if prediction == 1:
            return {"prediction": "Malade"}
        else:
            return {"prediction": "Sain"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


@app.post("/gpt-response", response_model=ChatResponse, tags=["Chat with GPT"], summary="Obtenir une réponse GPT", description="Obtenez une réponse du modèle GPT-3.5-turbo pour une question donnée.")
async def chat(request: ChatRequest):
    """
    Obtenez une réponse du modèle GPT-3.5-turbo pour une question donnée.

    - **message**: La question à poser au modèle GPT.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": request.message},
            ]
        )
        return {"response": completion.choices[0].message}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))