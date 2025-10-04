
# Vou criar um script para treinar os modelos e salvar os PKL + API endpoint
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("="*80)
print("CRIANDO MODELOS E PKL PARA API")
print("="*80)

# Carregar dados dos arquivos que funcionam (paste-2.txt e paste-3.txt)
print("\n1. Carregando dados...")
k2_df = pd.read_csv('paste-2.txt', comment='#', low_memory=False)
toi_df = pd.read_csv('paste-3.txt', comment='#', low_memory=False)

print(f"K2 shape: {k2_df.shape}")
print(f"TOI shape: {toi_df.shape}")

# K2 - Target e features
print("\n2. Processando K2...")
k2_target = k2_df['disposition'].map({
    'CONFIRMED': 1,
    'FALSE POSITIVE': 0,
    'CANDIDATE': 1
})

k2_features_list = ['pl_orbper', 'pl_rade', 'pl_trandep', 'st_teff', 'st_rad', 'st_logg']
k2_features = k2_df[k2_features_list].copy()

# Remover NaN
valid_k2 = ~(k2_features.isna().any(axis=1) | k2_target.isna())
k2_X = k2_features[valid_k2].reset_index(drop=True)
k2_y = k2_target[valid_k2].reset_index(drop=True)

print(f"K2 amostras válidas: {len(k2_X)}")
print(f"K2 target distribution:\n{k2_y.value_counts()}")

# TOI - Target e features
print("\n3. Processando TOI...")
toi_target = toi_df['tfopwg_disp'].map({
    'CP': 1, 'KP': 1, 'FP': 0, 'PC': 1, 'APC': 1, 'FA': 0
})

toi_features_list = ['pl_orbper', 'pl_rade', 'pl_trandep', 'st_teff', 'st_rad', 'st_logg']
toi_features = toi_df[toi_features_list].copy()

# Remover NaN
valid_toi = ~(toi_features.isna().any(axis=1) | toi_target.isna())
toi_X = toi_features[valid_toi].reset_index(drop=True)
toi_y = toi_target[valid_toi].reset_index(drop=True)

print(f"TOI amostras válidas: {len(toi_X)}")
print(f"TOI target distribution:\n{toi_y.value_counts()}")

# Combinar datasets
print("\n4. Combinando datasets...")
X = pd.concat([k2_X, toi_X], axis=0, ignore_index=True)
y = pd.concat([k2_y, toi_y], axis=0, ignore_index=True)

print(f"Dataset combinado: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split
print("\n5. Split treino/teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# Scaler
print("\n6. Escalonamento...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar XGBoost
print("\n7. Treinando XGBoost...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.5,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)
xgb_score = xgb_model.score(X_test_scaled, y_test)
print(f"XGBoost Test Accuracy: {xgb_score:.4f}")

# Treinar Random Forest
print("\n8. Treinando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
rf_score = rf_model.score(X_test_scaled, y_test)
print(f"Random Forest Test Accuracy: {rf_score:.4f}")

# Salvar modelos e scaler
print("\n9. Salvando modelos...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

joblib.dump(xgb_model, 'xgboost_exoplanet_model.pkl')
joblib.dump(rf_model, 'random_forest_exoplanet_model.pkl')
joblib.dump(scaler, 'scaler_exoplanet.pkl')

# Salvar metadata
metadata = {
    'feature_names': k2_features_list,
    'timestamp': timestamp,
    'xgb_accuracy': xgb_score,
    'rf_accuracy': rf_score,
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'target_map': {
        'K2': {'CONFIRMED': 1, 'FALSE POSITIVE': 0, 'CANDIDATE': 1},
        'TOI': {'CP': 1, 'KP': 1, 'FP': 0, 'PC': 1, 'APC': 1, 'FA': 0}
    }
}
joblib.dump(metadata, 'model_metadata.pkl')

print("\n✓ Modelos salvos:")
print("  - xgboost_exoplanet_model.pkl")
print("  - random_forest_exoplanet_model.pkl")
print("  - scaler_exoplanet.pkl")
print("  - model_metadata.pkl")

print(f"\nFeatures esperadas: {k2_features_list}")
