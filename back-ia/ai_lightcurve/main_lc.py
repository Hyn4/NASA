#!/usr/bin/env python3
"""
Script completo para treinamento de modelos de classificação de exoplanetas
usando dados de curvas de luz (lightcurve) do KELT Survey.

Inclui:
- Limpeza e preprocessamento de dados de lightcurve
- Feature engineering avançado
- Validação cruzada estratificada
- Bayesian optimization de hiperparâmetros
- Ensemble learning (Voting + Stacking)
- Explicabilidade com SHAP e LIME
- Treinamento com XGBoost e Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import os
import joblib
from glob import glob
from scipy import stats
from scipy.fft import fft, fftfreq

# Machine Learning
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    make_scorer
)
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier
)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost não disponível. Instale com: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Bayesian Optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-optimize não disponível. Usando RandomizedSearchCV.")
    from sklearn.model_selection import RandomizedSearchCV
    BAYESIAN_AVAILABLE = False

# SHAP e LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP não disponível. Instale com: pip install shap")
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️  LIME não disponível. Instale com: pip install lime")
    LIME_AVAILABLE = False

# Configurações
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Criar diretórios
os.makedirs('experiments/models', exist_ok=True)
os.makedirs('experiments/plots', exist_ok=True)
os.makedirs('experiments/results', exist_ok=True)
os.makedirs('experiments/shap', exist_ok=True)
os.makedirs('experiments/lime', exist_ok=True)


class LightcurveProcessor:
    """Processador para dados de curvas de luz do KELT"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.scaler = RobustScaler()
        self.feature_names = None
        
    def load_single_lightcurve(self, file_path):
        """Carrega uma única curva de luz do KELT"""
        try:
            df = pd.read_csv(
                file_path,
                delim_whitespace=True,
                comment='\\',
                names=['TIME', 'MAG', 'MAG_ERR'],
                skiprows=lambda x: x < 20
            )
            df = df.dropna()
            return df
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
            return None
    
    def extract_features_from_lightcurve(self, lc_data, file_name):
        """Extrai features estatísticas e de série temporal da curva de luz"""
        features = {}
        features['object_id'] = file_name
        features['n_observations'] = len(lc_data)
        
        if len(lc_data) < 10:
            return None
        
        mag = lc_data['MAG'].values
        features['mag_mean'] = np.mean(mag)
        features['mag_std'] = np.std(mag)
        features['mag_median'] = np.median(mag)
        features['mag_min'] = np.min(mag)
        features['mag_max'] = np.max(mag)
        features['mag_range'] = features['mag_max'] - features['mag_min']
        features['mag_var'] = np.var(mag)
        
        features['mag_p05'] = np.percentile(mag, 5)
        features['mag_p25'] = np.percentile(mag, 25)
        features['mag_p75'] = np.percentile(mag, 75)
        features['mag_p95'] = np.percentile(mag, 95)
        features['mag_iqr'] = features['mag_p75'] - features['mag_p25']
        
        features['mag_skew'] = stats.skew(mag)
        features['mag_kurtosis'] = stats.kurtosis(mag)
        
        features['mag_err_mean'] = np.mean(lc_data['MAG_ERR'].values)
        features['mag_err_std'] = np.std(lc_data['MAG_ERR'].values)
        
        features['snr'] = features['mag_std'] / features['mag_err_mean'] if features['mag_err_mean'] > 0 else 0
        features['norm_excess_var'] = (features['mag_std']**2 - features['mag_err_mean']**2) / features['mag_mean']**2 if features['mag_mean'] != 0 else 0
        
        time = lc_data['TIME'].values
        features['time_span'] = np.max(time) - np.min(time)
        features['time_mean_interval'] = np.mean(np.diff(time)) if len(time) > 1 else 0
        features['time_std_interval'] = np.std(np.diff(time)) if len(time) > 1 else 0
        
        mag_detrended = mag - np.median(mag)
        features['mag_detrend_std'] = np.std(mag_detrended)
        
        if len(mag) > 2:
            features['autocorr_lag1'] = np.corrcoef(mag[:-1], mag[1:])[0, 1]
        else:
            features['autocorr_lag1'] = 0
        
        try:
            if len(mag) > 10:
                fft_vals = np.abs(fft(mag_detrended))
                freqs = fftfreq(len(mag_detrended), d=features['time_mean_interval'])
                pos_mask = freqs > 0
                fft_pos = fft_vals[pos_mask]
                freqs_pos = freqs[pos_mask]
                
                if len(fft_pos) > 0:
                    features['fft_peak_power'] = np.max(fft_pos)
                    features['fft_peak_freq'] = freqs_pos[np.argmax(fft_pos)]
                    features['fft_total_power'] = np.sum(fft_pos**2)
                else:
                    features['fft_peak_power'] = 0
                    features['fft_peak_freq'] = 0
                    features['fft_total_power'] = 0
        except:
            features['fft_peak_power'] = 0
            features['fft_peak_freq'] = 0
            features['fft_total_power'] = 0
        
        mag_diff = np.diff(mag)
        features['mag_diff_mean'] = np.mean(mag_diff)
        features['mag_diff_std'] = np.std(mag_diff)
        features['mag_diff_max'] = np.max(np.abs(mag_diff))
        
        mag_mean = features['mag_mean']
        mag_std = features['mag_std']
        features['beyond_1std'] = np.sum(np.abs(mag - mag_mean) > mag_std) / len(mag)
        features['beyond_2std'] = np.sum(np.abs(mag - mag_mean) > 2 * mag_std) / len(mag)
        features['beyond_3std'] = np.sum(np.abs(mag - mag_mean) > 3 * mag_std) / len(mag)
        
        if len(mag) > 2:
            delta = (mag - features['mag_mean']) / features['mag_err_mean']
            features['stetson_j'] = np.sum(np.sign(delta[:-1]) * delta[:-1] * delta[1:]) / (len(mag) - 1)
            features['stetson_k'] = np.sum(np.abs(delta)) / np.sqrt(np.sum(delta**2)) / np.sqrt(len(mag))
        else:
            features['stetson_j'] = 0
            features['stetson_k'] = 0
        
        return features
    
    def load_all_lightcurves(self, max_files=None):
        """Carrega e processa todas as curvas de luz"""
        print("=" * 80)
        print("CARREGANDO CURVAS DE LUZ DO KELT")
        print("=" * 80)
        
        file_pattern = str(self.data_dir / "KELT_*.tbl")
        files = glob(file_pattern)
        
        if len(files) == 0:
            raise ValueError(f"Nenhum arquivo encontrado em {file_pattern}")
        
        print(f"\nEncontrados {len(files)} arquivos")
        
        if max_files:
            files = files[:max_files]
            print(f"Processando apenas {max_files} arquivos para teste")
        
        all_features = []
        
        for i, file_path in enumerate(files):
            if (i + 1) % 100 == 0:
                print(f"Processados {i + 1}/{len(files)} arquivos...")
            
            lc_data = self.load_single_lightcurve(file_path)
            
            if lc_data is None or len(lc_data) < 10:
                continue
            
            file_name = Path(file_path).stem
            features = self.extract_features_from_lightcurve(lc_data, file_name)
            
            if features is not None:
                all_features.append(features)
        
        print(f"\n✓ Processados {len(all_features)} objetos com sucesso")
        
        df = pd.DataFrame(all_features)
        
        print(f"\nShape do dataset: {df.shape}")
        print(f"\nPrimeiras colunas: {list(df.columns[:10])}")
        
        return df
    
    def create_synthetic_target(self, df):
        """Cria target sintético baseado em características de variabilidade"""
        print("\n" + "=" * 80)
        print("CRIANDO TARGET SINTÉTICO")
        print("=" * 80)
        
        std_norm = (df['mag_std'] - df['mag_std'].mean()) / df['mag_std'].std()
        fft_norm = (df['fft_peak_power'] - df['fft_peak_power'].mean()) / df['fft_peak_power'].std()
        snr_norm = (df['snr'] - df['snr'].mean()) / df['snr'].std()
        beyond3_norm = -(df['beyond_3std'] - df['beyond_3std'].mean()) / df['beyond_3std'].std()
        
        transit_score = std_norm + fft_norm + snr_norm + beyond3_norm
        threshold = np.percentile(transit_score, 70)
        target = (transit_score > threshold).astype(int)
        
        print(f"\nDistribuição do target:")
        print(pd.Series(target).value_counts())
        print(f"\nPercentual:")
        print(pd.Series(target).value_counts(normalize=True) * 100)
        
        print(f"\n⚠️  NOTA: Este é um target SINTÉTICO para demonstração.")
        print("Para uso real, substitua por labels verdadeiras de trânsitos confirmados.")
        
        return target


class AdvancedExoplanetClassifier:
    """Classificador avançado com Random Forest e XGBoost"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_rf_model = None
        self.best_xgb_model = None
        self.voting_model = None
        self.stacking_model = None
        self.scaler = RobustScaler()
        self.results = {}
        self.cv_results = {}
        self.shap_explainer = None
        self.lime_explainer = None
        
    def cross_validate_model(self, model, X, y, cv=5, model_name='Model'):
        """Validação cruzada estratificada com múltiplas métricas"""
        print(f"\n{'=' * 80}")
        print(f"VALIDAÇÃO CRUZADA - {model_name}")
        print(f"{'=' * 80}")
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_results = cross_validate(
            model, X, y,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        print(f"\nResultados da Validação Cruzada ({cv} folds):")
        print("-" * 60)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            
            print(f"\n{metric.upper()}:")
            print(f"  Train: {train_scores.mean():.4f} (±{train_scores.std():.4f})")
            print(f"  Test:  {test_scores.mean():.4f} (±{test_scores.std():.4f})")
        
        self.cv_results[model_name] = cv_results
        
        return cv_results
    
    def bayesian_optimize_rf(self, X_train, y_train, n_iter=30):
        """Otimização bayesiana de hiperparâmetros para Random Forest"""
        print(f"\n{'=' * 80}")
        print("OTIMIZAÇÃO BAYESIANA - RANDOM FOREST")
        print(f"{'=' * 80}")
        
        if BAYESIAN_AVAILABLE:
            search_space = {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(5, 30),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', None]),
                'bootstrap': Categorical([True, False])
            }
            
            opt = BayesSearchCV(
                RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                search_space,
                n_iter=n_iter,
                cv=5,
                scoring='f1',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        else:
            search_space = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [5, 10, 15, 20, 25, 30, None],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 6, 8, 10],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            opt = RandomizedSearchCV(
                RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                search_space,
                n_iter=n_iter,
                cv=5,
                scoring='f1',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        
        print(f"\nIniciando otimização com {n_iter} iterações...")
        opt.fit(X_train, y_train)
        
        print(f"\n✓ Otimização concluída!")
        print(f"\nMelhores hiperparâmetros:")
        for param, value in opt.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nMelhor F1-Score (CV): {opt.best_score_:.4f}")
        
        self.best_rf_model = opt.best_estimator_
        
        return opt.best_estimator_, opt.best_params_
    
    def bayesian_optimize_xgb(self, X_train, y_train, n_iter=30):
        """Otimização bayesiana para XGBoost"""
        if not XGBOOST_AVAILABLE:
            print("\n⚠️  XGBoost não disponível. Pulando otimização XGBoost.")
            return None, None
        
        print(f"\n{'=' * 80}")
        print("OTIMIZAÇÃO BAYESIANA - XGBOOST")
        print(f"{'=' * 80}")
        
        if BAYESIAN_AVAILABLE:
            search_space = {
                'n_estimators': Integer(100, 500),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 10),
                'min_child_weight': Integer(1, 10),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'gamma': Real(0.0, 0.5)
            }
            
            opt = BayesSearchCV(
                xgb.XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='logloss'),
                search_space,
                n_iter=n_iter,
                cv=5,
                scoring='f1',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        else:
            search_space = {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            }
            
            opt = RandomizedSearchCV(
                xgb.XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='logloss'),
                search_space,
                n_iter=n_iter,
                cv=5,
                scoring='f1',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        
        print(f"\nIniciando otimização com {n_iter} iterações...")
        opt.fit(X_train, y_train)
        
        print(f"\n✓ Otimização concluída!")
        print(f"\nMelhores hiperparâmetros:")
        for param, value in opt.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nMelhor F1-Score (CV): {opt.best_score_:.4f}")
        
        self.best_xgb_model = opt.best_estimator_
        
        return opt.best_estimator_, opt.best_params_
    
    def create_ensemble(self, X_train, y_train):
        """Cria modelos ensemble (Voting e Stacking)"""
        print(f"\n{'=' * 80}")
        print("CRIANDO ENSEMBLE MODELS")
        print(f"{'=' * 80}")
        
        if self.best_rf_model is None:
            raise ValueError("Execute bayesian_optimize_rf primeiro!")
        
        estimators = [('rf', self.best_rf_model)]
        
        if self.best_xgb_model is not None:
            estimators.append(('xgb', self.best_xgb_model))
        
        # Voting Classifier
        print("\n1. Voting Classifier (Soft Voting)")
        self.voting_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        self.voting_model.fit(X_train, y_train)
        print("✓ Voting Classifier treinado")
        
        # Stacking Classifier
        print("\n2. Stacking Classifier")
        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            cv=5,
            n_jobs=-1
        )
        
        self.stacking_model.fit(X_train, y_train)
        print("✓ Stacking Classifier treinado")
        
        return self.voting_model, self.stacking_model
    
    def evaluate_models(self, X_test, y_test):
        """Avalia todos os modelos no conjunto de teste"""
        print(f"\n{'=' * 80}")
        print("AVALIAÇÃO DOS MODELOS - CONJUNTO DE TESTE")
        print(f"{'=' * 80}")
        
        models = {
            'Random Forest': self.best_rf_model,
            'XGBoost': self.best_xgb_model,
            'Voting Ensemble': self.voting_model,
            'Stacking Ensemble': self.stacking_model
        }
        
        results_df = []
        
        for name, model in models.items():
            if model is None:
                continue
            
            print(f"\n{'-' * 60}")
            print(f"Modelo: {name}")
            print(f"{'-' * 60}")
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"AUC-ROC:   {auc:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            results_df.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'AUC-ROC': auc
            })
            
            self.results[name] = {
                'y_pred': y_pred,
                'y_proba': y_proba,
                'metrics': {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'auc_roc': auc
                }
            }
        
        results_df = pd.DataFrame(results_df)
        
        print(f"\n{'=' * 80}")
        print("COMPARAÇÃO FINAL DOS MODELOS")
        print(f"{'=' * 80}")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def explain_with_shap(self, X_train, X_test, feature_names, save_path='experiments/shap'):
        """Gera explicações SHAP para o melhor modelo individual (RF ou XGBoost)"""
        if not SHAP_AVAILABLE:
            print("\n⚠️  SHAP não está disponível")
            return
        
        print(f"\n{'=' * 80}")
        print("EXPLICABILIDADE COM SHAP")
        print(f"{'=' * 80}")
        
        # Usar o melhor modelo INDIVIDUAL (RF ou XGBoost, não ensemble)
        # SHAP TreeExplainer não suporta StackingClassifier
        if self.best_xgb_model is not None:
            model = self.best_xgb_model
            model_name = "XGBoost"
        elif self.best_rf_model is not None:
            model = self.best_rf_model
            model_name = "Random Forest"
        else:
            print("\n⚠️  Nenhum modelo individual disponível para SHAP")
            return
        
        print(f"\nUsando modelo: {model_name}")
        print("Criando explainer SHAP (pode demorar)...")
        
        try:
            # TreeExplainer para modelos baseados em árvore
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Se retornar lista (classificação binária), pegar classe 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            self.shap_explainer = explainer
            
            # 1. Summary Plot
            print("\n1. Gerando Summary Plot...")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(f"{save_path}/shap_summary_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Salvo em: {save_path}/shap_summary_plot.png")
            
            # 2. Feature Importance
            print("\n2. Gerando Feature Importance...")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig(f"{save_path}/shap_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Salvo em: {save_path}/shap_feature_importance.png")
            
            # 3. Dependence Plot para top 3 features
            print("\n3. Gerando Dependence Plots para top 3 features...")
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-3:]
            
            for idx in top_features_idx:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    idx, shap_values, X_test,
                    feature_names=feature_names,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(f"{save_path}/shap_dependence_{feature_names[idx]}.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Dependence plot para '{feature_names[idx]}' salvo")
            
            print(f"\n✓ Análise SHAP concluída!")
            
            return explainer, shap_values
        
        except Exception as e:
            print(f"\n❌ Erro ao gerar SHAP: {e}")
            return None, None
    
    def explain_with_lime(self, X_train, X_test, y_test, feature_names, n_samples=5, save_path='experiments/lime'):
        """Gera explicações LIME para amostras individuais"""
        if not LIME_AVAILABLE:
            print("\n⚠️  LIME não está disponível")
            return
        
        print(f"\n{'=' * 80}")
        print("EXPLICABILIDADE COM LIME")
        print(f"{'=' * 80}")
        
        # Usar melhor modelo individual
        if self.best_xgb_model is not None:
            model = self.best_xgb_model
        elif self.best_rf_model is not None:
            model = self.best_rf_model
        else:
            print("\n⚠️  Nenhum modelo disponível para LIME")
            return
        
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=['No Transit', 'Transit Candidate'],
            mode='classification',
            random_state=self.random_state
        )
        
        self.lime_explainer = explainer
        
        print(f"\nGerando explicações LIME para {n_samples} amostras...")
        
        y_pred = model.predict(X_test)
        correct_idx = np.where(y_pred == y_test)[0]
        incorrect_idx = np.where(y_pred != y_test)[0]
        
        samples_correct = min(n_samples // 2, len(correct_idx))
        samples_incorrect = min(n_samples - samples_correct, len(incorrect_idx))
        
        selected_idx = list(correct_idx[:samples_correct]) + list(incorrect_idx[:samples_incorrect])
        
        for i, idx in enumerate(selected_idx):
            print(f"\n{i+1}. Amostra {idx}:")
            print(f"   Verdadeiro: {y_test[idx]}, Predito: {y_pred[idx]}")
            
            exp = explainer.explain_instance(
                X_test[idx],
                model.predict_proba,
                num_features=10
            )
            
            exp.save_to_file(f"{save_path}/lime_sample_{idx}.html")
            print(f"   ✓ Explicação salva em: {save_path}/lime_sample_{idx}.html")
            
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            plt.savefig(f"{save_path}/lime_sample_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\n✓ Análise LIME concluída!")
        
        return explainer
    
    def save_models(self, save_path='experiments/models'):
        """Salva todos os modelos treinados"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\n{'=' * 80}")
        print("SALVANDO MODELOS")
        print(f"{'=' * 80}")
        
        models_to_save = {
            'random_forest': self.best_rf_model,
            'xgboost': self.best_xgb_model,
            'voting_ensemble': self.voting_model,
            'stacking_ensemble': self.stacking_model,
            'scaler': self.scaler
        }
        
        for name, model in models_to_save.items():
            if model is not None:
                file_path = f"{save_path}/{name}_model_{timestamp}.pkl"
                joblib.dump(model, file_path)
                print(f"✓ {name} salvo em: {file_path}")
        
        results_file = f"{save_path}/results_{timestamp}.pkl"
        joblib.dump(self.results, results_file)
        print(f"✓ Resultados salvos em: {results_file}")


def main():
    """Função principal"""
    print("=" * 80)
    print("TREINAMENTO AVANÇADO - CLASSIFICAÇÃO DE EXOPLANETAS")
    print("KELT Survey - Lightcurve Data")
    print("=" * 80)
    
    timestamp_start = datetime.now()
    print(f"\nInício: {timestamp_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configurações
    DATA_DIR = r"C:\Users\baron\Downloads\CODES\Hackatons\Nasa\AI\back-ia\ai_lightcurve\data\KELT_wget\output"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ITER_OPTIMIZATION = 30
    MAX_FILES = 1000
    
    try:
        # 1. Carregar e processar curvas de luz
        print("\n" + "=" * 80)
        print("ETAPA 1: PROCESSAMENTO DE CURVAS DE LUZ")
        print("=" * 80)
        
        processor = LightcurveProcessor(DATA_DIR)
        df = processor.load_all_lightcurves(max_files=MAX_FILES)
        
        # 2. Criar target
        target = processor.create_synthetic_target(df)
        
        # 3. Preparar features
        feature_cols = [col for col in df.columns if col != 'object_id']
        X = df[feature_cols].values
        y = target.values
        
        print(f"\n✓ Dataset preparado: {X.shape}")
        print(f"✓ Features: {len(feature_cols)}")
        print(f"✓ Target distribution: {np.bincount(y)}")
        
        # 4. Split train/test
        print("\n" + "=" * 80)
        print("ETAPA 2: SPLIT TREINO/TESTE")
        print("=" * 80)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"\nTrain shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"Train target: {np.bincount(y_train)}")
        print(f"Test target: {np.bincount(y_test)}")
        
        # 5. Escalar features
        print("\n" + "=" * 80)
        print("ETAPA 3: ESCALONAMENTO DE FEATURES")
        print("=" * 80)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("✓ Features escalonadas com RobustScaler")
        
        # 6. Treinamento avançado
        classifier = AdvancedExoplanetClassifier(random_state=RANDOM_STATE)
        classifier.scaler = scaler
        
        # 6.1 Validação Cruzada Inicial
        print("\n" + "=" * 80)
        print("ETAPA 4: VALIDAÇÃO CRUZADA INICIAL")
        print("=" * 80)
        
        rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        classifier.cross_validate_model(rf_baseline, X_train_scaled, y_train, cv=5, model_name='RF Baseline')
        
        # 6.2 Otimização Bayesiana
        print("\n" + "=" * 80)
        print("ETAPA 5: OTIMIZAÇÃO BAYESIANA DE HIPERPARÂMETROS")
        print("=" * 80)
        
        # Random Forest
        best_rf, best_rf_params = classifier.bayesian_optimize_rf(X_train_scaled, y_train, n_iter=N_ITER_OPTIMIZATION)
        
        # XGBoost
        best_xgb, best_xgb_params = classifier.bayesian_optimize_xgb(X_train_scaled, y_train, n_iter=N_ITER_OPTIMIZATION)
        
        # 6.3 Ensemble
        print("\n" + "=" * 80)
        print("ETAPA 6: CRIAÇÃO DE ENSEMBLE")
        print("=" * 80)
        
        voting_model, stacking_model = classifier.create_ensemble(X_train_scaled, y_train)
        
        # 7. Avaliação final
        print("\n" + "=" * 80)
        print("ETAPA 7: AVALIAÇÃO FINAL")
        print("=" * 80)
        
        results_df = classifier.evaluate_models(X_test_scaled, y_test)
        
        # 8. Explicabilidade
        print("\n" + "=" * 80)
        print("ETAPA 8: EXPLICABILIDADE (SHAP e LIME)")
        print("=" * 80)
        
        # SHAP (usando modelo individual, não ensemble)
        if SHAP_AVAILABLE:
            classifier.explain_with_shap(X_train_scaled, X_test_scaled, feature_cols)
        
        # LIME
        if LIME_AVAILABLE:
            classifier.explain_with_lime(X_train_scaled, X_test_scaled, y_test, feature_cols, n_samples=5)
        
        # 9. Salvar modelos
        print("\n" + "=" * 80)
        print("ETAPA 9: SALVANDO MODELOS")
        print("=" * 80)
        
        classifier.save_models()
        
        # 10. Resumo final
        timestamp_end = datetime.now()
        duration = timestamp_end - timestamp_start
        
        print("\n" + "=" * 80)
        print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 80)
        print(f"\nInício:   {timestamp_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Término:  {timestamp_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duração:  {duration}")
        
        print("\n" + "=" * 80)
        print("RESUMO DOS MELHORES RESULTADOS")
        print("=" * 80)
        
        best_model = results_df.loc[results_df['F1-Score'].idxmax()]
        print(f"\nMelhor Modelo: {best_model['Model']}")
        print(f"  Accuracy:  {best_model['Accuracy']:.4f}")
        print(f"  Precision: {best_model['Precision']:.4f}")
        print(f"  Recall:    {best_model['Recall']:.4f}")
        print(f"  F1-Score:  {best_model['F1-Score']:.4f}")
        print(f"  AUC-ROC:   {best_model['AUC-ROC']:.4f}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
