"""
Exemplos de uso da API de Classificação de Exoplanetas
"""
import requests
import json

# URL base da API
BASE_URL = "http://localhost:8000"


def test_health():
    """Testar health check"""
    print("\n" + "="*60)
    print("1. TESTE: Health Check")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Testar informações do modelo"""
    print("\n" + "="*60)
    print("2. TESTE: Model Info")
    print("="*60)

    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Testar predição única"""
    print("\n" + "="*60)
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

    print(f"\nInput Data:")
    print(json.dumps(data, indent=2))

    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    result = response.json()
    print(json.dumps(result, indent=2))

    print(f"\n📊 Resultado:")
    print(f"   Predição: {result['prediction_label']}")
    print(f"   Probabilidade: {result['probability']:.2%}")


def test_batch_prediction():
    """Testar predição em lote"""
    print("\n" + "="*60)
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

    print(f"\nInput Data: {len(data['exoplanets'])} exoplanetas")

    response = requests.post(f"{BASE_URL}/predict_batch", json=data)
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()

    print(f"\n📊 Resultados:")
    print(f"Total de predições: {result['total']}")

    for i, pred in enumerate(result['predictions'], 1):
        print(f"\n  Exoplaneta {i}:")
        print(f"    Predição: {pred['prediction_label']}")
        print(f"    Probabilidade: {pred['probability']:.2%}")


if __name__ == "__main__":
    print("="*60)
    print("TESTES DA API DE CLASSIFICAÇÃO DE EXOPLANETAS")
    print("="*60)
    print("\nCertifique-se de que a API está rodando em http://localhost:8000")
    print("Execute: uvicorn api:app --reload")

    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()

        print("\n" + "="*60)
        print("✓ TODOS OS TESTES CONCLUÍDOS!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERRO: Não foi possível conectar à API")
        print("Certifique-se de que a API está rodando!")
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
