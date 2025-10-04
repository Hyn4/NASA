
# Criar arquivo de requirements e documentação
requirements = '''# Requirements para API de Classificação de Exoplanetas
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
joblib==1.3.2
numpy==1.26.3
pandas==2.1.4
scikit-learn==1.4.0
python-multipart==0.0.6
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements)

print("✓ Requirements criado: requirements.txt")

# Criar README com instruções
readme = '''# API de Classificação de Exoplanetas 🪐

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
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
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
curl -X POST http://localhost:8000/predict_batch \\
  -H "Content-Type: application/json" \\
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
'''

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

print("✓ README criado: README.md")

# Criar exemplo de teste
test_example = '''"""
Exemplos de uso da API de Classificação de Exoplanetas
"""
import requests
import json

# URL base da API
BASE_URL = "http://localhost:8000"


def test_health():
    """Testar health check"""
    print("\\n" + "="*60)
    print("1. TESTE: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Testar informações do modelo"""
    print("\\n" + "="*60)
    print("2. TESTE: Model Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Testar predição única"""
    print("\\n" + "="*60)
    print("3. TESTE: Predição Única")
    print("="*60)
    
    # Exemplo de exoplaneta confirmado
    data = {
        "pl_orbper": 3.5,
        "pl_rade": 1.2,
        "pl_trandep": 1000.0,
        "st_teff": 5500.0,
        "st_rad": 1.0,
        "st_logg": 4.5
    }
    
    print(f"\\nInput Data:")
    print(json.dumps(data, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\\nStatus Code: {response.status_code}")
    print(f"Response:")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    print(f"\\n📊 Resultado:")
    print(f"   Predição: {result['prediction_label']}")
    print(f"   Probabilidade: {result['probability']:.2%}")


def test_batch_prediction():
    """Testar predição em lote"""
    print("\\n" + "="*60)
    print("4. TESTE: Predição em Lote")
    print("="*60)
    
    data = {
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
            },
            {
                "pl_orbper": 1.0,
                "pl_rade": 0.8,
                "pl_trandep": 500.0,
                "st_teff": 4500.0,
                "st_rad": 0.9,
                "st_logg": 4.6
            }
        ]
    }
    
    print(f"\\nInput Data: {len(data['exoplanets'])} exoplanetas")
    
    response = requests.post(f"{BASE_URL}/predict_batch", json=data)
    print(f"\\nStatus Code: {response.status_code}")
    result = response.json()
    
    print(f"\\n📊 Resultados:")
    print(f"Total de predições: {result['total']}")
    
    for i, pred in enumerate(result['predictions'], 1):
        print(f"\\n  Exoplaneta {i}:")
        print(f"    Predição: {pred['prediction_label']}")
        print(f"    Probabilidade: {pred['probability']:.2%}")


if __name__ == "__main__":
    print("="*60)
    print("TESTES DA API DE CLASSIFICAÇÃO DE EXOPLANETAS")
    print("="*60)
    print("\\nCertifique-se de que a API está rodando em http://localhost:8000")
    print("Execute: uvicorn api:app --reload")
    
    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        
        print("\\n" + "="*60)
        print("✓ TODOS OS TESTES CONCLUÍDOS!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\\n❌ ERRO: Não foi possível conectar à API")
        print("Certifique-se de que a API está rodando!")
    except Exception as e:
        print(f"\\n❌ ERRO: {e}")
'''

with open('test_api.py', 'w', encoding='utf-8') as f:
    f.write(test_example)

print("✓ Testes criados: test_api.py")

print("\n" + "="*80)
print("✅ TODOS OS ARQUIVOS CRIADOS COM SUCESSO!")
print("="*80)
print("\nArquivos disponíveis:")
print("  1. random_forest_exoplanet_model.pkl - Modelo treinado")
print("  2. scaler_exoplanet.pkl - Scaler")
print("  3. model_metadata.pkl - Metadados")
print("  4. api.py - Código da API FastAPI")
print("  5. requirements.txt - Dependências")
print("  6. README.md - Documentação")
print("  7. test_api.py - Testes da API")

print("\n📝 Para executar:")
print("  1. pip install -r requirements.txt")
print("  2. uvicorn api:app --reload")
print("  3. Acesse: http://localhost:8000/docs")
print("  4. Teste com: python test_api.py")
