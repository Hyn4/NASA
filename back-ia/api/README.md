# API de Classificação de Exoplanetas 🪐

API REST para classificação de exoplanetas usando Random Forest treinado com dados K2 e TOI da NASA.

## 📦 Arquivos

- `random_forest_exoplanet_model.pkl` - Modelo Random Forest treinado
- `scaler_exoplanet.pkl` - StandardScaler para normalização
- `model_metadata.pkl` - Metadados do modelo
- `api.py` - Código da API FastAPI
- `requirements.txt` - Dependências Python

## 🚀 Instalação

```bash
# Instalar dependências
pip install -r requirements.txt
```

## ▶️ Executar API

```bash
# Método 1: Usando uvicorn diretamente
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Método 2: Executar o script
python api.py
```

API estará disponível em: `http://localhost:8000`

## 📚 Documentação Interativa

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Informações do Modelo
```bash
curl http://localhost:8000/model_info
```

### 3. Predição Única
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pl_orbper": 3.5,
    "pl_rade": 1.2,
    "pl_trandep": 1000.0,
    "st_teff": 5500.0,
    "st_rad": 1.0,
    "st_logg": 4.5
  }'
```

### 4. Predição em Lote
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "exoplanets": [
      {
        "pl_orbper": 3.5,
        "pl_rade": 1.2,
        "pl_trandep": 1000.0,
        "st_teff": 5500.0,
        "st_rad": 1.0,
        "st_logg": 4.5
      },
      {
        "pl_orbper": 10.0,
        "pl_rade": 2.5,
        "pl_trandep": 2000.0,
        "st_teff": 6000.0,
        "st_rad": 1.2,
        "st_logg": 4.3
      }
    ]
  }'
```

## 📊 Features do Modelo

O modelo espera 6 features na seguinte ordem:

1. **pl_orbper** - Orbital Period (dias)
2. **pl_rade** - Planet Radius (Earth Radius)
3. **pl_trandep** - Transit Depth (ppm)
4. **st_teff** - Stellar Effective Temperature (K)
5. **st_rad** - Stellar Radius (Solar Radius)
6. **st_logg** - Stellar Surface Gravity (log10(cm/s**2))

## 📈 Performance do Modelo

- **Tipo**: RandomForestClassifier
- **Train Accuracy**: 96.67%
- **Test Accuracy**: 67.74%
- **Samples Train**: 120
- **Samples Test**: 31

## 🎯 Resposta da API

```json
{
  "prediction": 1,
  "prediction_label": "Confirmed Planet",
  "probability": 0.85,
  "probabilities": {
    "not_confirmed": 0.15,
    "confirmed": 0.85
  },
  "timestamp": "2025-10-04T04:08:56.123456"
}
```

## 🐍 Exemplo em Python

```python
import requests

# URL da API
url = "http://localhost:8000/predict"

# Dados do exoplaneta
data = {
    "pl_orbper": 3.5,
    "pl_rade": 1.2,
    "pl_trandep": 1000.0,
    "st_teff": 5500.0,
    "st_rad": 1.0,
    "st_logg": 4.5
}

# Fazer requisição
response = requests.post(url, json=data)
result = response.json()

print(f"Predição: {result['prediction_label']}")
print(f"Probabilidade: {result['probability']:.2%}")
```

## 🌐 Deploy (Opcional)

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY *.pkl .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build e Run
```bash
docker build -t exoplanet-api .
docker run -p 8000:8000 exoplanet-api
```

## 📝 Notas

- Todas as features devem ser numéricas
- Valores ausentes não são permitidos
- Features são automaticamente escalonadas pelo StandardScaler
- Modelo retorna probabilidades para ambas as classes

## 📄 Licença

MIT License
