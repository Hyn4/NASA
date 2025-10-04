
# Criar código da API FastAPI
api_code = '''
"""
API REST para Classificação de Exoplanetas
Modelo: Random Forest treinado com datasets K2 e TOI da NASA

Endpoints:
- POST /predict - Predição única
- POST /predict_batch - Predição em lote
- GET /health - Health check
- GET /model_info - Informações do modelo
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Inicializar FastAPI
app = FastAPI(
    title="Exoplanet Classification API",
    description="API para classificação de exoplanetas usando Random Forest",
    version="1.0.0"
)

# Carregar modelos e scaler
try:
    model = joblib.load('random_forest_exoplanet_model.pkl')
    scaler = joblib.load('scaler_exoplanet.pkl')
    metadata = joblib.load('model_metadata.pkl')
    print("✓ Modelos carregados com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelos: {e}")
    model = None
    scaler = None
    metadata = None


# Schemas Pydantic
class ExoplanetFeatures(BaseModel):
    """Features de entrada para predição"""
    pl_orbper: float = Field(..., description="Orbital Period (days)", example=3.5)
    pl_rade: float = Field(..., description="Planet Radius (Earth Radius)", example=1.2)
    pl_trandep: float = Field(..., description="Transit Depth (ppm)", example=1000.0)
    st_teff: float = Field(..., description="Stellar Effective Temperature (K)", example=5500.0)
    st_rad: float = Field(..., description="Stellar Radius (Solar Radius)", example=1.0)
    st_logg: float = Field(..., description="Stellar Surface Gravity (log10(cm/s**2))", example=4.5)
    
    class Config:
        schema_extra = {
            "example": {
                "pl_orbper": 3.5,
                "pl_rade": 1.2,
                "pl_trandep": 1000.0,
                "st_teff": 5500.0,
                "st_rad": 1.0,
                "st_logg": 4.5
            }
        }


class PredictionResponse(BaseModel):
    """Resposta da predição"""
    prediction: int = Field(..., description="0 = Not Confirmed, 1 = Confirmed")
    prediction_label: str = Field(..., description="Label descritivo da predição")
    probability: float = Field(..., description="Probabilidade da classe predita")
    probabilities: dict = Field(..., description="Probabilidades de todas as classes")
    timestamp: str = Field(..., description="Timestamp da predição")


class BatchPredictionRequest(BaseModel):
    """Request para predições em lote"""
    exoplanets: List[ExoplanetFeatures]


class BatchPredictionResponse(BaseModel):
    """Resposta de predições em lote"""
    predictions: List[PredictionResponse]
    total: int


class ModelInfo(BaseModel):
    """Informações do modelo"""
    model_type: str
    feature_names: List[str]
    train_accuracy: float
    test_accuracy: float
    n_samples_train: int
    n_samples_test: int
    timestamp: str


# Endpoints
@app.get("/")
def root():
    """Endpoint raiz"""
    return {
        "message": "Exoplanet Classification API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predição única",
            "POST /predict_batch": "Predição em lote",
            "GET /health": "Health check",
            "GET /model_info": "Informações do modelo"
        }
    }


@app.get("/health")
def health_check():
    """Health check da API"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model_info", response_model=ModelInfo)
def get_model_info():
    """Retorna informações sobre o modelo"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata não carregado")
    
    return ModelInfo(
        model_type=metadata['model_type'],
        feature_names=metadata['feature_names'],
        train_accuracy=metadata['train_accuracy'],
        test_accuracy=metadata['test_accuracy'],
        n_samples_train=metadata['n_samples_train'],
        n_samples_test=metadata['n_samples_test'],
        timestamp=metadata['timestamp']
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ExoplanetFeatures):
    """
    Predição de classificação de exoplaneta
    
    Args:
        features: Features do exoplaneta
        
    Returns:
        PredictionResponse com predição e probabilidades
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")
    
    try:
        # Converter para array na ordem correta
        feature_array = np.array([[
            features.pl_orbper,
            features.pl_rade,
            features.pl_trandep,
            features.st_teff,
            features.st_rad,
            features.st_logg
        ]])
        
        # Escalar features
        feature_scaled = scaler.transform(feature_array)
        
        # Predição
        prediction = model.predict(feature_scaled)[0]
        probabilities = model.predict_proba(feature_scaled)[0]
        
        # Label descritivo
        prediction_label = "Confirmed Planet" if prediction == 1 else "Not Confirmed (False Positive)"
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            probability=float(probabilities[prediction]),
            probabilities={
                "not_confirmed": float(probabilities[0]),
                "confirmed": float(probabilities[1])
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Predição em lote de múltiplos exoplanetas
    
    Args:
        request: Lista de features de exoplanetas
        
    Returns:
        BatchPredictionResponse com todas as predições
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")
    
    predictions = []
    
    for features in request.exoplanets:
        try:
            # Converter para array
            feature_array = np.array([[
                features.pl_orbper,
                features.pl_rade,
                features.pl_trandep,
                features.st_teff,
                features.st_rad,
                features.st_logg
            ]])
            
            # Escalar e predizer
            feature_scaled = scaler.transform(feature_array)
            prediction = model.predict(feature_scaled)[0]
            probabilities = model.predict_proba(feature_scaled)[0]
            
            prediction_label = "Confirmed Planet" if prediction == 1 else "Not Confirmed (False Positive)"
            
            predictions.append(PredictionResponse(
                prediction=int(prediction),
                prediction_label=prediction_label,
                probability=float(probabilities[prediction]),
                probabilities={
                    "not_confirmed": float(probabilities[0]),
                    "confirmed": float(probabilities[1])
                },
                timestamp=datetime.now().isoformat()
            ))
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {str(e)}")
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions)
    )


# Executar com: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

# Salvar arquivo da API
with open('api.py', 'w', encoding='utf-8') as f:
    f.write(api_code)

print("✓ API criada: api.py")
