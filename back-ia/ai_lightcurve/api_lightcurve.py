#!/usr/bin/env python3
"""
Script de teste para API de classificação de lightcurves
"""

import requests
import json

# URL da API
BASE_URL = "http://localhost:8000"


def test_health():
    """Testa health check"""
    print("\n" + "=" * 80)
    print("1. TESTE: Health Check")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def test_list_models():
    """Testa listagem de modelos"""
    print("\n" + "=" * 80)
    print("2. TESTE: Listar Modelos")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def test_single_prediction(file_path):
    """Testa predição única"""
    print("\n" + "=" * 80)
    print("3. TESTE: Predição Única")
    print("=" * 80)
    
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/plain')}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 RESULTADO:")
        print(f"  Arquivo: {file_path}")
        print(f"  Predição: {result['prediction_label']}")
        print(f"  Confiança: {result['confidence']:.2%}")
        print(f"  Modelo usado: {result['model_used']}")
        print(f"\n  Probabilidades:")
        print(f"    No Transit: {result['probabilities']['no_transit']:.2%}")
        print(f"    Transit Candidate: {result['probabilities']['transit_candidate']:.2%}")
        
        print(f"\n  Features extraídas (primeiras 10):")
        for i, (key, value) in enumerate(list(result['features'].items())[:10]):
            print(f"    {key}: {value:.6f}")
    else:
        print(f"Erro: {response.text}")


def test_batch_prediction(file_paths):
    """Testa predição em lote"""
    print("\n" + "=" * 80)
    print("4. TESTE: Predição em Lote")
    print("=" * 80)
    
    files = []
    for file_path in file_paths:
        files.append(('files', (file_path, open(file_path, 'rb'), 'text/plain')))
    
    response = requests.post(f"{BASE_URL}/predict_batch", files=files)
    
    # Fechar arquivos
    for _, (_, f, _) in files:
        f.close()
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 RESULTADO:")
        print(f"  Total de arquivos: {result['total_files']}")
        print(f"  Sucessos: {result['successful']}")
        print(f"  Falhas: {result['failed']}")
        
        if result['results']:
            print(f"\n  Predições:")
            for res in result['results']:
                print(f"    • {res['filename']}: {res['prediction_label']} ({res['confidence']:.2%})")
    else:
        print(f"Erro: {response.text}")


if __name__ == "__main__":
    print("=" * 80)
    print("TESTES DA API DE CLASSIFICAÇÃO DE LIGHTCURVES")
    print("=" * 80)
    print("\nCertifique-se de que a API está rodando em http://localhost:8000")
    print("Execute: uvicorn api_lightcurve:app --reload")
    
    try:
        # Testes
        test_health()
        test_list_models()
        
        # Ajuste o caminho para seu arquivo de teste
        test_file = "path/to/your/KELT_N02_lc_000001_V01_east_tfa_lc.tbl"
        
        # Teste com arquivo único
        # test_single_prediction(test_file)
        
        # Teste com múltiplos arquivos
        # test_batch_prediction([test_file, test_file])
        
        print("\n" + "=" * 80)
        print("✓ TESTES CONCLUÍDOS!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERRO: Não foi possível conectar à API")
        print("Certifique-se de que a API está rodando!")
    except FileNotFoundError:
        print("\n❌ ERRO: Arquivo de teste não encontrado")
        print("Ajuste o caminho do arquivo no script")
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
