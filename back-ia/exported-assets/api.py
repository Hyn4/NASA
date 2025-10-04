"""
API REST para Classificação de Exoplanetas
Modelo: Random Forest treinado com datasets K2 e TOI da NASA

Endpoints:
- POST /predict - Predição única com features
- POST /predict_batch - Predição em lote com features
- POST /predict_lightcurve - Predição a partir de arquivo lightcurve
- POST /parse_json - Upload de JSON
- GET /health - Health check
- GET /model_info - Informações do modelo
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.fft import fft, fftfreq
import json
from pathlib import Path
from glob import glob

# Inicializar FastAPI
app = FastAPI(
    title="Exoplanet Classification API",
    description="API para classificação de exoplanetas usando Random Forest",
    version="2.0.0"
)

# Definir diretórios
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "experiments" / "models"

print("=" * 80)
print("CARREGANDO MODELOS")
print("=" * 80)
print(f"Diretório base: {BASE_DIR}")
print(f"Diretório de modelos: {MODELS_DIR}")

# Função para encontrar o modelo mais recente
def load_latest_model():
    """Carrega o modelo, scaler e metadata mais recentes"""
    global model, scaler, metadata
    
    try:
        # Verificar se diretório existe
        if not MODELS_DIR.exists():
            print(f"❌ Diretório não encontrado: {MODELS_DIR}")
            return None, None, None
        
        # Listar arquivos .pkl
        all_pkl_files = list(MODELS_DIR.glob("*.pkl"))
        print(f"\n✓ Encontrados {len(all_pkl_files)} arquivos .pkl")
        
        if len(all_pkl_files) == 0:
            print("❌ Nenhum arquivo .pkl encontrado!")
            return None, None, None
        
        # Buscar modelos específicos
        model_files = sorted(MODELS_DIR.glob("*random_forest*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        scaler_files = sorted(MODELS_DIR.glob("*scaler*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Carregar modelo
        if model_files:
            model_path = model_files[0]
            model = joblib.load(model_path)
            print(f"✓ Modelo carregado: {model_path.name}")
        else:
            print("❌ Nenhum modelo Random Forest encontrado")
            model = None
        
        # Carregar scaler
        if scaler_files:
            scaler_path = scaler_files[0]
            scaler = joblib.load(scaler_path)
            print(f"✓ Scaler carregado: {scaler_path.name}")
        else:
            print("❌ Nenhum scaler encontrado")
            scaler = None
        
        # Tentar carregar metadata (opcional)
        metadata_files = sorted(MODELS_DIR.glob("*metadata*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        if metadata_files:
            metadata_path = metadata_files[0]
            metadata = joblib.load(metadata_path)
            print(f"✓ Metadata carregado: {metadata_path.name}")
        else:
            print("⚠️  Metadata não encontrado (opcional)")
            # Criar metadata básico
            metadata = {
                'model_type': 'Random Forest',
                'feature_names': ['feature_' + str(i) for i in range(35)],
                'train_accuracy': 0.0,
                'test_accuracy': 0.0,
                'n_samples_train': 0,
                'n_samples_test': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        print("=" * 80)
        return model, scaler, metadata
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Carregar modelos ao iniciar
model, scaler, metadata = load_latest_model()

# Feature names para lightcurve
LIGHTCURVE_FEATURE_NAMES = [
    'n_observations', 'mag_mean', 'mag_std', 'mag_median', 'mag_min',
    'mag_max', 'mag_range', 'mag_var', 'mag_p05', 'mag_p25',
    'mag_p75', 'mag_p95', 'mag_iqr', 'mag_skew', 'mag_kurtosis',
    'mag_err_mean', 'mag_err_std', 'snr', 'norm_excess_var',
    'time_span', 'time_mean_interval', 'time_std_interval',
    'mag_detrend_std', 'autocorr_lag1', 'fft_peak_power',
    'fft_peak_freq', 'fft_total_power', 'mag_diff_mean',
    'mag_diff_std', 'mag_diff_max', 'beyond_1std', 'beyond_2std',
    'beyond_3std', 'stetson_j', 'stetson_k'
]


# Schemas Pydantic
class ExoplanetFeatures(BaseModel):
    """Features de entrada para predição"""
    pl_orbper: float = Field(..., description="Orbital Period (days)", example=3.5)
    pl_rade: float = Field(..., description="Planet Radius (Earth Radius)", example=1.2)
    pl_trandep: float = Field(..., description="Transit Depth (ppm)", example=1000.0)
    st_teff: float = Field(..., description="Stellar Effective Temperature (K)", example=5500.0)
    st_rad: float = Field(..., description="Stellar Radius (Solar Radius)", example=1.0)
    st_logg: float = Field(..., description="Stellar Surface Gravity (log10(cm/s**2))", example=4.5)

    model_config = {
        "json_schema_extra": {
            "example": {
                "pl_orbper": 3.5,
                "pl_rade": 1.2,
                "pl_trandep": 1000.0,
                "st_teff": 5500.0,
                "st_rad": 1.0,
                "st_logg": 4.5
            }
        }
    }


class PredictionResponse(BaseModel):
    """Resposta da predição"""
    prediction: int = Field(..., description="0 = Not Confirmed, 1 = Confirmed")
    prediction_label: str = Field(..., description="Label descritivo da predição")
    probability: float = Field(..., description="Probabilidade da classe predita")
    probabilities: dict = Field(..., description="Probabilidades de todas as classes")
    timestamp: str = Field(..., description="Timestamp da predição")


class LightcurvePredictionResponse(BaseModel):
    """Resposta da predição com lightcurve"""
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    features_extracted: Dict[str, float]
    n_observations: int
    filename: str
    timestamp: str


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


# Processador de Lightcurve
class LightcurveProcessor:
    """Classe para processar arquivos de lightcurve"""
    
    @staticmethod
    def load_lightcurve_from_content(content: str) -> pd.DataFrame:
        """Carrega lightcurve de string/conteúdo do arquivo"""
        try:
            lines = content.strip().split('\n')
            data = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('\\') and not line.startswith('|'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            time = float(parts[0])
                            mag = float(parts[1])
                            mag_err = float(parts[2])
                            data.append([time, mag, mag_err])
                        except ValueError:
                            continue
            
            if len(data) == 0:
                raise ValueError("Nenhum dado válido encontrado no arquivo")
            
            df = pd.DataFrame(data, columns=['TIME', 'MAG', 'MAG_ERR'])
            return df.dropna()
        
        except Exception as e:
            raise ValueError(f"Erro ao processar lightcurve: {str(e)}")
    
    @staticmethod
    def extract_features(lc_data: pd.DataFrame) -> Dict[str, float]:
        """Extrai features estatísticas da lightcurve"""
        if len(lc_data) < 10:
            raise ValueError("Lightcurve precisa ter pelo menos 10 observações")
        
        features = {}
        mag = lc_data['MAG'].values
        
        # Estatísticas básicas
        features['mag_mean'] = float(np.mean(mag))
        features['mag_std'] = float(np.std(mag))
        features['mag_median'] = float(np.median(mag))
        features['mag_min'] = float(np.min(mag))
        features['mag_max'] = float(np.max(mag))
        features['mag_range'] = features['mag_max'] - features['mag_min']
        features['mag_var'] = float(np.var(mag))
        
        # Percentis
        features['mag_p05'] = float(np.percentile(mag, 5))
        features['mag_p25'] = float(np.percentile(mag, 25))
        features['mag_p75'] = float(np.percentile(mag, 75))
        features['mag_p95'] = float(np.percentile(mag, 95))
        features['mag_iqr'] = features['mag_p75'] - features['mag_p25']
        
        # Skewness e Kurtosis
        features['mag_skew'] = float(stats.skew(mag))
        features['mag_kurtosis'] = float(stats.kurtosis(mag))
        
        # Erro
        features['mag_err_mean'] = float(np.mean(lc_data['MAG_ERR'].values))
        features['mag_err_std'] = float(np.std(lc_data['MAG_ERR'].values))
        
        # SNR
        features['snr'] = features['mag_std'] / features['mag_err_mean'] if features['mag_err_mean'] > 0 else 0
        
        # Variabilidade normalizada
        features['norm_excess_var'] = (features['mag_std']**2 - features['mag_err_mean']**2) / features['mag_mean']**2 if features['mag_mean'] != 0 else 0
        
        # Features temporais
        time = lc_data['TIME'].values
        features['time_span'] = float(np.max(time) - np.min(time))
        features['time_mean_interval'] = float(np.mean(np.diff(time))) if len(time) > 1 else 0
        features['time_std_interval'] = float(np.std(np.diff(time))) if len(time) > 1 else 0
        
        # Magnitude detrended
        mag_detrended = mag - np.median(mag)
        features['mag_detrend_std'] = float(np.std(mag_detrended))
        
        # Autocorrelação
        if len(mag) > 2:
            features['autocorr_lag1'] = float(np.corrcoef(mag[:-1], mag[1:])[0, 1])
        else:
            features['autocorr_lag1'] = 0
        
        # FFT features
        try:
            if len(mag) > 10:
                fft_vals = np.abs(fft(mag_detrended))
                freqs = fftfreq(len(mag_detrended), d=features['time_mean_interval'])
                pos_mask = freqs > 0
                fft_pos = fft_vals[pos_mask]
                freqs_pos = freqs[pos_mask]
                
                if len(fft_pos) > 0:
                    features['fft_peak_power'] = float(np.max(fft_pos))
                    features['fft_peak_freq'] = float(freqs_pos[np.argmax(fft_pos)])
                    features['fft_total_power'] = float(np.sum(fft_pos**2))
                else:
                    features['fft_peak_power'] = 0
                    features['fft_peak_freq'] = 0
                    features['fft_total_power'] = 0
        except:
            features['fft_peak_power'] = 0
            features['fft_peak_freq'] = 0
            features['fft_total_power'] = 0
        
        # Diferenças consecutivas
        mag_diff = np.diff(mag)
        features['mag_diff_mean'] = float(np.mean(mag_diff))
        features['mag_diff_std'] = float(np.std(mag_diff))
        features['mag_diff_max'] = float(np.max(np.abs(mag_diff)))
        
        # Outliers
        mag_mean = features['mag_mean']
        mag_std = features['mag_std']
        features['beyond_1std'] = float(np.sum(np.abs(mag - mag_mean) > mag_std) / len(mag))
        features['beyond_2std'] = float(np.sum(np.abs(mag - mag_mean) > 2 * mag_std) / len(mag))
        features['beyond_3std'] = float(np.sum(np.abs(mag - mag_mean) > 3 * mag_std) / len(mag))
        
        # Stetson J e K
        if len(mag) > 2:
            delta = (mag - features['mag_mean']) / features['mag_err_mean']
            features['stetson_j'] = float(np.sum(np.sign(delta[:-1]) * delta[:-1] * delta[1:]) / (len(mag) - 1))
            features['stetson_k'] = float(np.sum(np.abs(delta)) / np.sqrt(np.sum(delta**2)) / np.sqrt(len(mag)))
        else:
            features['stetson_j'] = 0
            features['stetson_k'] = 0
        
        features['n_observations'] = len(lc_data)
        
        return features


# Endpoints
@app.get("/")
def root():
    """Endpoint raiz"""
    return {
        "message": "Exoplanet Classification API",
        "version": "2.0.0",
        "models_loaded": model is not None and scaler is not None,
        "endpoints": {
            "POST /predict": "Predição única com features",
            "POST /predict_batch": "Predição em lote com features",
            "POST /predict_lightcurve": "Predição a partir de arquivo lightcurve (.tbl)",
            "POST /parse_json": "Upload de JSON com múltiplos exoplanetas",
            "GET /health": "Health check",
            "GET /model_info": "Informações do modelo"
        }
    }


@app.get("/health")
def health_check():
    """Health check da API"""
    if model is None or scaler is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "scaler_loaded": False,
            "error": "Modelos não carregados. Verifique o diretório: " + str(MODELS_DIR),
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "models_directory": str(MODELS_DIR),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model_info", response_model=ModelInfo)
def get_model_info():
    """Retorna informações sobre o modelo"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata não carregado")
    
    return ModelInfo(
        model_type=metadata.get('model_type', 'Random Forest'),
        feature_names=metadata.get('feature_names', []),
        train_accuracy=metadata.get('train_accuracy', 0.0),
        test_accuracy=metadata.get('test_accuracy', 0.0),
        n_samples_train=metadata.get('n_samples_train', 0),
        n_samples_test=metadata.get('n_samples_test', 0),
        timestamp=metadata.get('timestamp', datetime.now().isoformat())
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ExoplanetFeatures):
    """Predição com features diretas"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")
    
    try:
        feature_array = np.array([[
            features.pl_orbper,
            features.pl_rade,
            features.pl_trandep,
            features.st_teff,
            features.st_rad,
            features.st_logg
        ]])
        
        feature_scaled = scaler.transform(feature_array)
        prediction = model.predict(feature_scaled)[0]
        probabilities = model.predict_proba(feature_scaled)[0]
        
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
    """Predição em lote"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")
    
    predictions = []
    
    for features in request.exoplanets:
        try:
            feature_array = np.array([[
                features.pl_orbper,
                features.pl_rade,
                features.pl_trandep,
                features.st_teff,
                features.st_rad,
                features.st_logg
            ]])
            
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
            raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions)
    )


@app.post("/predict_lightcurve", response_model=LightcurvePredictionResponse)
async def predict_lightcurve(file: UploadFile = File(...)):
    """Classifica exoplaneta a partir de arquivo lightcurve (.tbl)"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")
    
    if not file.filename.endswith(('.tbl', '.txt', '.dat')):
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Use .tbl, .txt ou .dat"
        )
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        processor = LightcurveProcessor()
        lc_data = processor.load_lightcurve_from_content(content_str)
        features = processor.extract_features(lc_data)
        
        feature_array = np.array([features.get(name, 0) for name in LIGHTCURVE_FEATURE_NAMES]).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)
        
        prediction = int(model.predict(feature_scaled)[0])
        probabilities = model.predict_proba(feature_scaled)[0]
        
        prediction_label = "Transit Candidate" if prediction == 1 else "No Transit"
        confidence = float(probabilities[prediction])
        
        return LightcurvePredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            probabilities={
                "no_transit": float(probabilities[0]),
                "transit_candidate": float(probabilities[1])
            },
            features_extracted=features,
            n_observations=len(lc_data),
            filename=file.filename,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
    finally:
        await file.close()


@app.post("/parse_json", response_model=BatchPredictionResponse)
async def parse_json(file: UploadFile = File(...)):
    """Upload de JSON com múltiplos exoplanetas"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")
    
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser .json")
    
    try:
        content = await file.read()
        json_data = json.loads(content.decode('utf-8'))
        
        if "exoplanets" not in json_data:
            raise HTTPException(
                status_code=400,
                detail="JSON deve conter chave 'exoplanets'"
            )
        
        request = BatchPredictionRequest(exoplanets=json_data["exoplanets"])
        
        predictions = []
        
        for idx, features in enumerate(request.exoplanets):
            try:
                feature_array = np.array([[
                    features.pl_orbper,
                    features.pl_rade,
                    features.pl_trandep,
                    features.st_teff,
                    features.st_rad,
                    features.st_logg
                ]])
                
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
                raise HTTPException(
                    status_code=500,
                    detail=f"Erro no exoplaneta {idx + 1}: {str(e)}"
                )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions)
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {str(e)}")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Erro de validação: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
    finally:
        await file.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
