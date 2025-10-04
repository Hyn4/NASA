
# Vou criar apenas com Random Forest já que XGBoost não está disponível
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*80)
print("CRIANDO MODELOS E PKL PARA API")
print("="*80)

# Carregar dados
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

print(f"\nDataset combinado: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Percentual:\n{y.value_counts(normalize=True)*100}")

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

# Treinar Random Forest
print("\n7. Treinando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Avaliar
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\nRandom Forest Train Accuracy: {train_acc:.4f}")
print(f"Random Forest Test Accuracy: {test_acc:.4f}")

print(f"\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test, target_names=['Not Confirmed', 'Confirmed']))

print(f"\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test))

# Feature importance
print(f"\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': k2_features_list,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.to_string(index=False))

# Salvar modelos e scaler
print("\n8. Salvando modelos...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

joblib.dump(rf_model, 'random_forest_exoplanet_model.pkl')
joblib.dump(scaler, 'scaler_exoplanet.pkl')

# Salvar metadata
metadata = {
    'feature_names': k2_features_list,
    'timestamp': timestamp,
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'model_type': 'RandomForestClassifier',
    'target_encoding': {
        '0': 'Not Confirmed (False Positive)',
        '1': 'Confirmed (Confirmed Planet/Candidate)'
    },
    'target_map': {
        'K2': {'CONFIRMED': 1, 'FALSE POSITIVE': 0, 'CANDIDATE': 1},
        'TOI': {'CP': 1, 'KP': 1, 'FP': 0, 'PC': 1, 'APC': 1, 'FA': 0}
    }
}
joblib.dump(metadata, 'model_metadata.pkl')

print("\n" + "="*80)
print("✓ MODELOS SALVOS COM SUCESSO!")
print("="*80)
print("\nArquivos criados:")
print("  1. random_forest_exoplanet_model.pkl")
print("  2. scaler_exoplanet.pkl")
print("  3. model_metadata.pkl")

print(f"\n\nFeatures esperadas pela API (em ordem):")
for i, feat in enumerate(k2_features_list, 1):
    print(f"  {i}. {feat}")

print(f"\nModelo treinado em: {timestamp}")
print(f"Acurácia de treino: {train_acc:.4f}")
print(f"Acurácia de teste: {test_acc:.4f}")
