from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from models import Model
app = FastAPI()

model = Model()
model.eval()

class PredictionResponse(BaseModel):
    fecha: str
    prediccion: float

class Top5TokenResponse(BaseModel):
    categoria: str
    tokens: Dict[str, List[PredictionResponse]]

@app.get("/predict_token/{token}/{days}", response_model=List[PredictionResponse])
def get_token_prediction(token: str, days: int):
    try:
        df = model.df[model.df['symbol'] == token]
        if df.empty:
            raise HTTPException(status_code=404, detail="Token no encontrado")
        
        predictions = model.predict_by_token(days, token, df)
        predictions = [{"fecha": pred["fecha"].isoformat(), "prediccion": pred["prediccion"]} for pred in predictions]
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/predict_category/{category}/{days}", response_model=Dict[str, List[PredictionResponse]])
def get_category_prediction(category: str, days: int):
    try:
        predictions_by_category = model.predict_by_category(days, category)
        response = {
            token: [{"fecha": pred["fecha"].isoformat(), "prediccion": pred["prediccion"]} for pred in predictions.to_dict(orient='records')]
            for token, predictions in predictions_by_category.items()
        }
        return response
    except ValueError:
        raise HTTPException(status_code=404, detail="Categoría no encontrada")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top5_tokens", response_model=List[Top5TokenResponse])
def get_top_5_tokens():
    try:
        categories = ["gaming", "meme", "ai", "rwa"]
        top_5_per_category = []
        
        for category in categories:
            top_tokens = model.top_5_tokens_by_category(category, days=30)
            predictions = {}
            for token, _ in top_tokens:
                df = model.get_dfcategories(category)
                token_predictions = model.predict_by_token(30, token, df)
                predictions[token] = [{"fecha": pred["fecha"].isoformat(), "prediccion": pred["prediccion"]} for pred in token_predictions]
            top_5_per_category.append({
                "categoria": category,
                "tokens": predictions
            })
        return top_5_per_category
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/')
def index():
    return {"message": "API de predicción de tokens"}