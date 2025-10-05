"""
API REST para Classificação de Exoplanetas e Cálculo de Similaridade com a Terra
Modelo: Random Forest treinado com datasets K2 e TOI da NASA
ESI Model: Earth Similarity Index Calculator

Endpoints:
- POST /predict - Predição única
- POST /predict_batch - Predição em lote
- POST /earth_similarity - Cálculo de similaridade com a Terra
- POST /earth_similarity_batch - Cálculo de similaridade em lote
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
from earth_similarity import EarthSimilarityCalculator

# Inicializar FastAPI
app = FastAPI(
    title="Exoplanet Classification & Earth Similarity API",
    description="API para classificação de exoplanetas e cálculo de similaridade com a Terra",
    version="1.0.0"
)

# Carregar modelos e scaler
try:
    # Modelos de classificação
    model = joblib.load('random_forest_exoplanet_model.pkl')
    scaler = joblib.load('scaler_exoplanet.pkl')
    metadata = joblib.load('model_metadata.pkl')
    
    # Modelo de similaridade com a Terra
    esi_model = joblib.load('earth_similarity_model.pkl')
    
    print("✓ Modelos carregados com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelos: {e}")
    model = None
    scaler = None
    metadata = None
    esi_model = None

# Schemas Pydantic para Classificação
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

# Schemas Pydantic para Similaridade com a Terra
class EarthSimilarityFeatures(BaseModel):
    """Features para cálculo do Earth Similarity Index"""
    pl_bmasse: float = Field(..., description="Planet Mass (Earth Mass)", example=1.1)
    pl_rade: float = Field(..., description="Planet Radius (Earth Radius)", example=1.05)
    pl_orbper: float = Field(..., description="Orbital Period (days)", example=367.0)
    st_teff: float = Field(..., description="Stellar Effective Temperature (K)", example=5800.0)
    st_rad: float = Field(..., description="Stellar Radius (Solar Radius)", example=1.02)

    class Config:
        schema_extra = {
            "example": {
                "pl_bmasse": 1.1,
                "pl_rade": 1.05,
                "pl_orbper": 367.0,
                "st_teff": 5800.0,
                "st_rad": 1.02
            }
        }

class EarthSimilarityResponse(BaseModel):
    """Resposta do cálculo de similaridade"""
    earth_similarity_index: float = Field(..., description="Earth Similarity Index (0-1)")
    similarity_category: str = Field(..., description="Categoria de similaridade")
    timestamp: str = Field(..., description="Timestamp do cálculo")

class BatchSimilarityRequest(BaseModel):
    """Request para cálculos de similaridade em lote"""
    planets: List[EarthSimilarityFeatures]

class BatchSimilarityResponse(BaseModel):
    """Resposta de cálculos de similaridade em lote"""
    results: List[EarthSimilarityResponse]
    total: int
    most_similar: EarthSimilarityResponse

class ModelInfo(BaseModel):
    """Informações do modelo"""
    model_type: str
    feature_names: List[str]
    train_accuracy: float
    test_accuracy: float
    n_samples_train: int
    n_samples_test: int
    timestamp: str

# Função auxiliar para categorizar similaridade
def get_similarity_category(esi_value: float) -> str:
    """Categoriza o valor ESI em faixas de similaridade"""
    if esi_value >= 0.9:
        return "Extremely Similar to Earth"
    elif esi_value >= 0.8:
        return "Very Similar to Earth"
    elif esi_value >= 0.6:
        return "Moderately Similar to Earth"
    elif esi_value >= 0.4:
        return "Somewhat Similar to Earth"
    elif esi_value >= 0.2:
        return "Low Similarity to Earth"
    else:
        return "Very Low Similarity to Earth"

# Endpoints
@app.get("/")
def root():
    """Endpoint raiz"""
    return {
        "message": "Exoplanet Classification & Earth Similarity API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predição única de classificação",
            "POST /predict_batch": "Predição em lote de classificação",
            "POST /earth_similarity": "Cálculo de similaridade com a Terra",
            "POST /earth_similarity_batch": "Cálculo de similaridade em lote",
            "GET /health": "Health check",
            "GET /model_info": "Informações do modelo"
        }
    }

@app.get("/health")
def health_check():
    """Health check da API"""
    if model is None or scaler is None or esi_model is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")

    return {
        "status": "healthy",
        "classification_model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "esi_model_loaded": esi_model is not None,
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
        raise HTTPException(status_code=503, detail="Modelos de classificação não carregados")

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
        raise HTTPException(status_code=503, detail="Modelos de classificação não carregados")

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

@app.post("/earth_similarity", response_model=EarthSimilarityResponse)
def calculate_earth_similarity(features: EarthSimilarityFeatures):
    """
    Calcula o Earth Similarity Index de um exoplaneta

    Args:
        features: Features do exoplaneta para cálculo de similaridade

    Returns:
        EarthSimilarityResponse com ESI e categoria
    """
    if esi_model is None:
        raise HTTPException(status_code=503, detail="Modelo ESI não carregado")

    try:
        # Calcular ESI
        esi_value = esi_model.calculate_single_planet(
            pl_bmasse=features.pl_bmasse,
            pl_rade=features.pl_rade,
            pl_orbper=features.pl_orbper,
            st_teff=features.st_teff,
            st_rad=features.st_rad
        )

        # Categorizar similaridade
        similarity_category = get_similarity_category(esi_value)

        return EarthSimilarityResponse(
            earth_similarity_index=float(esi_value),
            similarity_category=similarity_category,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no cálculo de similaridade: {str(e)}")

@app.post("/earth_similarity_batch", response_model=BatchSimilarityResponse)
def calculate_earth_similarity_batch(request: BatchSimilarityRequest):
    """
    Calcula o Earth Similarity Index para múltiplos exoplanetas

    Args:
        request: Lista de features de exoplanetas

    Returns:
        BatchSimilarityResponse com todos os cálculos de similaridade
    """
    if esi_model is None:
        raise HTTPException(status_code=503, detail="Modelo ESI não carregado")

    results = []

    for features in request.planets:
        try:
            # Calcular ESI
            esi_value = esi_model.calculate_single_planet(
                pl_bmasse=features.pl_bmasse,
                pl_rade=features.pl_rade,
                pl_orbper=features.pl_orbper,
                st_teff=features.st_teff,
                st_rad=features.st_rad
            )

            # Categorizar similaridade
            similarity_category = get_similarity_category(esi_value)

            result = EarthSimilarityResponse(
                earth_similarity_index=float(esi_value),
                similarity_category=similarity_category,
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no cálculo de similaridade em lote: {str(e)}")

    # Encontrar o mais similar
    most_similar = max(results, key=lambda x: x.earth_similarity_index)

    return BatchSimilarityResponse(
        results=results,
        total=len(results),
        most_similar=most_similar
    )

# Executar com: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)