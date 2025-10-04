#!/usr/bin/env python3
"""
Script completo para treinamento de modelos de classificação de exoplanetas
usando datasets K2 e TOI do NASA Exoplanet Archive.

Modelos:
- XGBoost para classificação
- Random Forest para feature importance

Inclui:
- Limpeza e preprocessamento de dados
- Feature engineering
- Treinamento com validação cruzada
- Avaliação de métricas
- Feature importance
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

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Configurações
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Criar diretórios
os.makedirs('experiments/models', exist_ok=True)
os.makedirs('experiments/plots', exist_ok=True)
os.makedirs('experiments/results', exist_ok=True)


class ExoplanetDataProcessor:
    """Processador para datasets de exoplanetas K2 e TOI"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self, k2_path, toi_path):
        """
        Carrega datasets K2 e TOI
        
        Args:
            k2_path: Caminho para arquivo K2
            toi_path: Caminho para arquivo TOI
        """
        print("=" * 80)
        print("CARREGANDO DATASETS")
        print("=" * 80)
        
        # Carregar K2
        print(f"\nCarregando K2 dataset: {k2_path}")
        k2_df = pd.read_csv(k2_path, comment='#', low_memory=False)
        print(f"K2 shape: {k2_df.shape}")
        
        # Carregar TOI
        print(f"\nCarregando TOI dataset: {toi_path}")
        toi_df = pd.read_csv(toi_path, comment='#', low_memory=False)
        print(f"TOI shape: {toi_df.shape}")
        
        return k2_df, toi_df
    
    def clean_data(self, df, dataset_name='Dataset'):
        """
        Limpeza de dados
        
        Args:
            df: DataFrame a ser limpo
            dataset_name: Nome do dataset para logging
        """
        print(f"\n{'=' * 80}")
        print(f"LIMPEZA DE DADOS - {dataset_name}")
        print(f"{'=' * 80}")
        
        initial_shape = df.shape
        print(f"\nShape inicial: {initial_shape}")
        
        # Remover colunas com mais de 80% de valores ausentes
        threshold = 0.8
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        print(f"\nRemovendo {len(cols_to_drop)} colunas com >80% missing:")
        if cols_to_drop:
            print(f"Colunas removidas: {cols_to_drop[:10]}..." if len(cols_to_drop) > 10 else f"Colunas removidas: {cols_to_drop}")
        
        df = df.drop(columns=cols_to_drop)
        
        # Converter colunas numéricas
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        print(f"Shape após limpeza: {df.shape}")
        print(f"Colunas removidas: {initial_shape[1] - df.shape[1]}")
        
        return df
    
    def create_features(self, df, dataset_name='K2'):
        """
        Feature engineering
        
        Args:
            df: DataFrame com dados limpos
            dataset_name: Nome do dataset ('K2' ou 'TOI')
        """
        print(f"\n{'=' * 80}")
        print(f"FEATURE ENGINEERING - {dataset_name}")
        print(f"{'=' * 80}")
        
        df_features = df.copy()
        
        # Features comuns entre K2 e TOI
        common_features = []
        
        # Orbital features
        if 'pl_orbper' in df_features.columns:
            common_features.append('pl_orbper')
        
        if 'pl_orbsmax' in df_features.columns:
            common_features.append('pl_orbsmax')
        
        if 'pl_orbeccen' in df_features.columns:
            common_features.append('pl_orbeccen')
            
        if 'pl_orbincl' in df_features.columns:
            common_features.append('pl_orbincl')
        
        # Planet physical properties
        if 'pl_rade' in df_features.columns:
            common_features.append('pl_rade')
            
        if 'pl_radj' in df_features.columns:
            common_features.append('pl_radj')
            
        if 'pl_masse' in df_features.columns:
            common_features.append('pl_masse')
            
        if 'pl_massj' in df_features.columns:
            common_features.append('pl_massj')
            
        if 'pl_dens' in df_features.columns:
            common_features.append('pl_dens')
        
        # Transit features
        if 'pl_trandep' in df_features.columns:
            common_features.append('pl_trandep')
            
        if 'pl_trandur' in df_features.columns:
            common_features.append('pl_trandur')
        
        # Environmental features
        if 'pl_insol' in df_features.columns:
            common_features.append('pl_insol')
            
        if 'pl_eqt' in df_features.columns:
            common_features.append('pl_eqt')
        
        # Stellar properties
        if 'st_teff' in df_features.columns:
            common_features.append('st_teff')
            
        if 'st_rad' in df_features.columns:
            common_features.append('st_rad')
            
        if 'st_mass' in df_features.columns:
            common_features.append('st_mass')
            
        if 'st_logg' in df_features.columns:
            common_features.append('st_logg')
            
        if 'st_met' in df_features.columns:
            common_features.append('st_met')
        
        # Distance and magnitude
        if 'sy_dist' in df_features.columns:
            common_features.append('sy_dist')
            
        if dataset_name == 'TOI' and 'st_tmag' in df_features.columns:
            common_features.append('st_tmag')
        
        # Derived features
        if 'pl_rade' in df_features.columns and 'pl_masse' in df_features.columns:
            df_features['planet_density_derived'] = (
                df_features['pl_masse'] / (df_features['pl_rade'] ** 3)
            )
            common_features.append('planet_density_derived')
        
        if 'pl_orbper' in df_features.columns and 'pl_orbsmax' in df_features.columns:
            df_features['orbital_velocity'] = (
                2 * np.pi * df_features['pl_orbsmax'] / df_features['pl_orbper']
            )
            common_features.append('orbital_velocity')
        
        if 'st_rad' in df_features.columns and 'st_mass' in df_features.columns:
            df_features['stellar_density'] = (
                df_features['st_mass'] / (df_features['st_rad'] ** 3)
            )
            common_features.append('stellar_density')
        
        # Selecionar apenas features existentes
        existing_features = [f for f in common_features if f in df_features.columns]
        
        print(f"\nFeatures selecionadas: {len(existing_features)}")
        print(f"Features: {existing_features}")
        
        return df_features, existing_features
    
    def create_target(self, df, dataset_name='K2'):
        """
        Cria variável target baseada em disposition ou tfopwg_disp
        Estratégias alternativas se não houver coluna disponível
        
        Args:
            df: DataFrame
            dataset_name: 'K2' ou 'TOI'
        """
        print(f"\n{'=' * 80}")
        print(f"CRIANDO TARGET - {dataset_name}")
        print(f"{'=' * 80}")
        
        target = None
        valid_indices = None
        
        if dataset_name == 'K2':
            # K2 - Tentar múltiplas estratégias
            if 'disposition' in df.columns:
                print(f"\nTentando usar coluna 'disposition'...")
                print(f"Distribuição ORIGINAL:")
                print(df['disposition'].value_counts(dropna=False))
                
                # Verificar se tem valores não-NaN
                non_null_count = df['disposition'].notna().sum()
                
                if non_null_count > 0:
                    target_col = 'disposition'
                    target = df[target_col].copy()
                    
                    target_map = {
                        'CONFIRMED': 1,
                        'FALSE POSITIVE': 0,
                        'CANDIDATE': 1
                    }
                    
                    target = target.map(target_map)
                    valid_indices = target.notna()
                    
                    print(f"✓ Usando 'disposition' com {valid_indices.sum()} amostras válidas")
                else:
                    print(f"✗ Coluna 'disposition' está toda NaN!")
            
            # Estratégia alternativa: usar default_flag ou outras colunas
            if target is None or valid_indices is None or valid_indices.sum() == 0:
                print(f"\n⚠️  AVISO: Não há target válido em 'disposition'!")
                print(f"Tentando estratégia alternativa baseada em 'default_flag'...")
                
                if 'default_flag' in df.columns:
                    target = df['default_flag'].copy()
                    valid_indices = target.notna()
                    print(f"✓ Usando 'default_flag' com {valid_indices.sum()} amostras válidas")
                else:
                    # Última alternativa: criar target sintético baseado em features
                    print(f"\n⚠️  Criando target sintético baseado em features disponíveis...")
                    # Usar pl_rade como threshold: planetas > 2 Earth radii = 1, menores = 0
                    if 'pl_rade' in df.columns:
                        target = (df['pl_rade'] > 2).astype(int)
                        valid_indices = df['pl_rade'].notna()
                        print(f"✓ Target sintético criado baseado em pl_rade (threshold=2 Earth radii)")
                        print(f"  Amostras válidas: {valid_indices.sum()}")
                    else:
                        raise ValueError("Não foi possível criar target para K2!")
        
        else:  # TOI
            if 'tfopwg_disp' in df.columns:
                print(f"\nTentando usar coluna 'tfopwg_disp'...")
                print(f"Distribuição ORIGINAL:")
                print(df['tfopwg_disp'].value_counts(dropna=False))
                
                non_null_count = df['tfopwg_disp'].notna().sum()
                
                if non_null_count > 0:
                    target_col = 'tfopwg_disp'
                    target = df[target_col].copy()
                    
                    target_map = {
                        'CP': 1,
                        'KP': 1,
                        'FP': 0,
                        'PC': 1,
                        'APC': 1,
                        'FA': 0
                    }
                    
                    target = target.map(target_map)
                    valid_indices = target.notna()
                    
                    print(f"✓ Usando 'tfopwg_disp' com {valid_indices.sum()} amostras válidas")
                else:
                    print(f"✗ Coluna 'tfopwg_disp' está toda NaN!")
            
            # Estratégia alternativa para TOI
            if target is None or valid_indices is None or valid_indices.sum() == 0:
                print(f"\n⚠️  AVISO: Não há target válido em 'tfopwg_disp'!")
                print(f"Tentando estratégia alternativa baseado em features...")
                
                # Usar pl_rade como threshold
                if 'pl_rade' in df.columns:
                    target = (df['pl_rade'] > 2).astype(int)
                    valid_indices = df['pl_rade'].notna()
                    print(f"✓ Target sintético criado baseado em pl_rade (threshold=2 Earth radii)")
                    print(f"  Amostras válidas: {valid_indices.sum()}")
                else:
                    raise ValueError("Não foi possível criar target para TOI!")
        
        print(f"\nDistribuição FINAL do target:")
        if valid_indices is not None and valid_indices.sum() > 0:
            print(target[valid_indices].value_counts())
            print(f"\nPercentual:")
            print(target[valid_indices].value_counts(normalize=True) * 100)
            print(f"\nTotal de amostras válidas: {valid_indices.sum()}")
        else:
            print("NENHUMA amostra válida encontrada!")
        
        return target, valid_indices
    
    def prepare_dataset(self, k2_path, toi_path, use_dataset='both'):
        """
        Pipeline completo de preparação de dados
        
        Args:
            k2_path: Caminho para K2
            toi_path: Caminho para TOI
            use_dataset: 'k2', 'toi' ou 'both'
        """
        print("\n" + "=" * 80)
        print("PIPELINE DE PREPARAÇÃO DE DADOS")
        print("=" * 80)
        
        # Carregar dados
        k2_df, toi_df = self.load_data(k2_path, toi_path)
        
        datasets = []
        
        if use_dataset in ['k2', 'both']:
            try:
                # Processar K2
                k2_clean = self.clean_data(k2_df, 'K2')
                
                # Criar target ANTES de criar features
                k2_target, k2_valid = self.create_target(k2_clean, 'K2')
                
                # Criar features
                k2_features_df, k2_feature_names = self.create_features(k2_clean, 'K2')
                
                # Alinhar features e target
                k2_features = k2_features_df[k2_feature_names].copy()
                
                print(f"\nK2 - Shape das features: {k2_features.shape}")
                print(f"K2 - Shape do target: {k2_target.shape}")
                print(f"K2 - Índices válidos: {k2_valid.sum()}")
                
                if k2_valid.sum() > 0:
                    datasets.append(('K2', k2_features, k2_target, k2_valid))
                else:
                    print(f"⚠️  K2 dataset não tem amostras válidas - pulando")
            except Exception as e:
                print(f"⚠️  Erro ao processar K2: {e}")
        
        if use_dataset in ['toi', 'both']:
            try:
                # Processar TOI
                toi_clean = self.clean_data(toi_df, 'TOI')
                
                # Criar target ANTES de criar features
                toi_target, toi_valid = self.create_target(toi_clean, 'TOI')
                
                # Criar features
                toi_features_df, toi_feature_names = self.create_features(toi_clean, 'TOI')
                
                # Alinhar features e target
                toi_features = toi_features_df[toi_feature_names].copy()
                
                print(f"\nTOI - Shape das features: {toi_features.shape}")
                print(f"TOI - Shape do target: {toi_target.shape}")
                print(f"TOI - Índices válidos: {toi_valid.sum()}")
                
                if toi_valid.sum() > 0:
                    datasets.append(('TOI', toi_features, toi_target, toi_valid))
                else:
                    print(f"⚠️  TOI dataset não tem amostras válidas - pulando")
            except Exception as e:
                print(f"⚠️  Erro ao processar TOI: {e}")
        
        if len(datasets) == 0:
            raise ValueError("Nenhum dataset válido encontrado! Verifique seus arquivos de dados.")
        
        # Combinar datasets se necessário
        if use_dataset == 'both' and len(datasets) == 2:
            print(f"\n{'=' * 80}")
            print("COMBINANDO DATASETS K2 E TOI")
            print(f"{'=' * 80}")
            
            k2_name, k2_features, k2_target, k2_valid = datasets[0]
            toi_name, toi_features, toi_target, toi_valid = datasets[1]
            
            # Encontrar features comuns
            common_features = list(set(k2_features.columns) & set(toi_features.columns))
            print(f"\nFeatures comuns entre K2 e TOI: {len(common_features)}")
            print(f"Features: {common_features}")
            
            if len(common_features) == 0:
                print("⚠️  Nenhuma feature comum - usando apenas o primeiro dataset disponível")
                name, features, target, valid = datasets[0]
                X = features[valid].reset_index(drop=True)
                y = target[valid].reset_index(drop=True)
                self.feature_names = list(features.columns)
            else:
                # Selecionar apenas features comuns
                k2_features_common = k2_features[common_features].copy()
                toi_features_common = toi_features[common_features].copy()
                
                # Aplicar máscara de índices válidos
                k2_features_common = k2_features_common[k2_valid].reset_index(drop=True)
                k2_target_valid = k2_target[k2_valid].reset_index(drop=True)
                
                toi_features_common = toi_features_common[toi_valid].reset_index(drop=True)
                toi_target_valid = toi_target[toi_valid].reset_index(drop=True)
                
                print(f"\nK2 após filtrar índices válidos: {k2_features_common.shape}")
                print(f"TOI após filtrar índices válidos: {toi_features_common.shape}")
                
                # Concatenar
                X = pd.concat([k2_features_common, toi_features_common], axis=0, ignore_index=True)
                y = pd.concat([k2_target_valid, toi_target_valid], axis=0, ignore_index=True)
                
                self.feature_names = common_features
            
        elif len(datasets) == 1:
            name, features, target, valid = datasets[0]
            X = features[valid].reset_index(drop=True)
            y = target[valid].reset_index(drop=True)
            self.feature_names = list(features.columns)
            print(f"\nUsando apenas dataset: {name}")
        else:
            # Usar o primeiro dataset com mais amostras
            datasets_sorted = sorted(datasets, key=lambda x: x[3].sum(), reverse=True)
            name, features, target, valid = datasets_sorted[0]
            X = features[valid].reset_index(drop=True)
            y = target[valid].reset_index(drop=True)
            self.feature_names = list(features.columns)
            print(f"\nUsando dataset com mais amostras: {name}")
        
        # Remover linhas com NaN nas features
        print(f"\n{'=' * 80}")
        print(f"REMOÇÃO FINAL DE NaN")
        print(f"{'=' * 80}")
        print(f"\nShape antes de remover NaN: X={X.shape}, y={y.shape}")
        
        # Verificar NaN em cada coluna
        nan_counts = X.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\nNaN por feature:")
            for feat, count in nan_counts[nan_counts > 0].items():
                print(f"  {feat}: {count} ({count/len(X)*100:.1f}%)")
        
        valid_mask = ~(X.isna().any(axis=1))
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        
        print(f"\nDataset final:")
        print(f"Shape: {X.shape}")
        print(f"Target distribution:")
        print(y.value_counts())
        print(f"\nPercentual:")
        print(y.value_counts(normalize=True) * 100)
        
        if len(X) == 0:
            raise ValueError("Dataset final está vazio! Verifique os dados e o mapeamento do target.")
        
        return X, y


class ExoplanetClassifier:
    """Classificador de exoplanetas usando XGBoost e Random Forest"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.xgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Treina modelo XGBoost"""
        print(f"\n{'=' * 80}")
        print("TREINANDO XGBOOST")
        print(f"{'=' * 80}")
        
        # Parâmetros XGBoost ajustados para datasets pequenos
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
            'random_state': self.random_state,
            'eval_metric': 'logloss'
        }
        
        print(f"\nParâmetros: {params}")
        
        # Treinar
        self.xgb_model = xgb.XGBClassifier(**params)
        
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Predições
        y_pred_train = self.xgb_model.predict(X_train)
        y_pred_val = self.xgb_model.predict(X_val)
        y_proba_val = self.xgb_model.predict_proba(X_val)[:, 1]
        
        # Métricas
        train_metrics = self._calculate_metrics(y_train, y_pred_train, None)
        val_metrics = self._calculate_metrics(y_val, y_pred_val, y_proba_val)
        
        print(f"\n{'=' * 40}")
        print("RESULTADOS XGBOOST - TREINO")
        print(f"{'=' * 40}")
        self._print_metrics(train_metrics)
        
        print(f"\n{'=' * 40}")
        print("RESULTADOS XGBOOST - VALIDAÇÃO")
        print(f"{'=' * 40}")
        self._print_metrics(val_metrics)
        
        self.results['xgboost'] = {
            'train': train_metrics,
            'val': val_metrics
        }
        
        return self.xgb_model
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Treina modelo Random Forest"""
        print(f"\n{'=' * 80}")
        print("TREINANDO RANDOM FOREST")
        print(f"{'=' * 80}")
        
        # Parâmetros Random Forest ajustados
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': 0
        }
        
        print(f"\nParâmetros: {params}")
        
        # Treinar
        self.rf_model = RandomForestClassifier(**params)
        self.rf_model.fit(X_train, y_train)
        
        # Predições
        y_pred_train = self.rf_model.predict(X_train)
        y_pred_val = self.rf_model.predict(X_val)
        y_proba_val = self.rf_model.predict_proba(X_val)[:, 1]
        
        # Métricas
        train_metrics = self._calculate_metrics(y_train, y_pred_train, None)
        val_metrics = self._calculate_metrics(y_val, y_pred_val, y_proba_val)
        
        print(f"\n{'=' * 40}")
        print("RESULTADOS RANDOM FOREST - TREINO")
        print(f"{'=' * 40}")
        self._print_metrics(train_metrics)
        
        print(f"\n{'=' * 40}")
        print("RESULTADOS RANDOM FOREST - VALIDAÇÃO")
        print(f"{'=' * 40}")
        self._print_metrics(val_metrics)
        
        self.results['random_forest'] = {
            'train': train_metrics,
            'val': val_metrics
        }
        
        return self.rf_model
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calcula métricas de avaliação"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['auc_roc'] = None
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Imprime métricas"""
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        if 'auc_roc' in metrics and metrics['auc_roc'] is not None:
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
    
    def plot_feature_importance(self, feature_names, save_path='experiments/plots'):
        """Plota feature importance para ambos os modelos"""
        print(f"\n{'=' * 80}")
        print("FEATURE IMPORTANCE")
        print(f"{'=' * 80}")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        xgb_df = None
        rf_df = None
        
        # XGBoost feature importance
        if self.xgb_model is not None:
            xgb_importance = self.xgb_model.feature_importances_
            xgb_df = pd.DataFrame({
                'feature': feature_names,
                'importance': xgb_importance
            }).sort_values('importance', ascending=False)
            
            top_n = min(20, len(xgb_df))
            xgb_top = xgb_df.head(top_n)
            
            axes[0].barh(range(len(xgb_top)), xgb_top['importance'])
            axes[0].set_yticks(range(len(xgb_top)))
            axes[0].set_yticklabels(xgb_top['feature'])
            axes[0].set_xlabel('Importance')
            axes[0].set_title(f'XGBoost - Top {top_n} Features')
            axes[0].invert_yaxis()
            
            print("\nXGBoost - Top Features:")
            print(xgb_df.to_string(index=False))
        
        # Random Forest feature importance
        if self.rf_model is not None:
            rf_importance = self.rf_model.feature_importances_
            rf_df = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_importance
            }).sort_values('importance', ascending=False)
            
            top_n = min(20, len(rf_df))
            rf_top = rf_df.head(top_n)
            
            axes[1].barh(range(len(rf_top)), rf_top['importance'])
            axes[1].set_yticks(range(len(rf_top)))
            axes[1].set_yticklabels(rf_top['feature'])
            axes[1].set_xlabel('Importance')
            axes[1].set_title(f'Random Forest - Top {top_n} Features')
            axes[1].invert_yaxis()
            
            print("\nRandom Forest - Top Features:")
            print(rf_df.to_string(index=False))
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_file = f"{save_path}/feature_importance_{timestamp}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\nFeature importance salvo em: {save_file}")
        plt.close()
        
        return xgb_df, rf_df
    
    def plot_confusion_matrices(self, X_val, y_val, save_path='experiments/plots'):
        """Plota confusion matrices"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        if self.xgb_model is not None:
            y_pred_xgb = self.xgb_model.predict(X_val)
            cm_xgb = confusion_matrix(y_val, y_pred_xgb)
            
            sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title('XGBoost - Confusion Matrix')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')
        
        if self.rf_model is not None:
            y_pred_rf = self.rf_model.predict(X_val)
            cm_rf = confusion_matrix(y_val, y_pred_rf)
            
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
            axes[1].set_title('Random Forest - Confusion Matrix')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        save_file = f"{save_path}/confusion_matrices_{timestamp}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices salvas em: {save_file}")
        plt.close()
    
    def save_models(self, save_path='experiments/models'):
        """Salva modelos treinados"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.xgb_model is not None:
            xgb_file = f"{save_path}/xgboost_model_{timestamp}.pkl"
            joblib.dump(self.xgb_model, xgb_file)
            print(f"\nXGBoost salvo em: {xgb_file}")
        
        if self.rf_model is not None:
            rf_file = f"{save_path}/random_forest_model_{timestamp}.pkl"
            joblib.dump(self.rf_model, rf_file)
            print(f"Random Forest salvo em: {rf_file}")


def main():
    """Função principal"""
    print("=" * 80)
    print("TREINAMENTO DE MODELOS - CLASSIFICAÇÃO DE EXOPLANETAS")
    print("NASA Exoplanet Archive - K2 e TOI Datasets")
    print("=" * 80)
    
    timestamp_start = datetime.now()
    print(f"\nInício: {timestamp_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configurações
    K2_PATH = 'k2.csv'
    TOI_PATH = 'TOI.csv'
    USE_DATASET = 'both'  # 'k2', 'toi' ou 'both'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    try:
        # 1. Processar dados
        processor = ExoplanetDataProcessor()
        X, y = processor.prepare_dataset(K2_PATH, TOI_PATH, use_dataset=USE_DATASET)
        
        # Verificar se temos dados suficientes
        if len(X) < 20:
            print(f"\n⚠️  AVISO: Dataset muito pequeno ({len(X)} amostras)!")
            print("Recomenda-se ter pelo menos 50 amostras para treinamento adequado.")
        
        # 2. Split train/val
        print(f"\n{'=' * 80}")
        print("SPLIT TREINO/VALIDAÇÃO")
        print(f"{'=' * 80}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"\nTrain shape: {X_train.shape}")
        print(f"Val shape: {X_val.shape}")
        print(f"\nTrain target distribution:")
        print(y_train.value_counts())
        print(f"\nVal target distribution:")
        print(y_val.value_counts())
        
        # 3. Escalar features
        print(f"\n{'=' * 80}")
        print("ESCALONAMENTO DE FEATURES")
        print(f"{'=' * 80}")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Converter de volta para DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=processor.feature_names)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=processor.feature_names)
        
        print("\nFeatures escalonadas com StandardScaler")
        
        # 4. Treinar modelos
        classifier = ExoplanetClassifier(random_state=RANDOM_STATE)
        
        # XGBoost
        xgb_model = classifier.train_xgboost(X_train_scaled, y_train.values, X_val_scaled, y_val.values)
        
        # Random Forest
        rf_model = classifier.train_random_forest(X_train_scaled, y_train.values, X_val_scaled, y_val.values)
        
        # 5. Feature Importance
        xgb_importance, rf_importance = classifier.plot_feature_importance(processor.feature_names)
        
        # 6. Confusion Matrices
        classifier.plot_confusion_matrices(X_val_scaled, y_val.values)
        
        # 7. Salvar modelos
        classifier.save_models()
        
        # 8. Resumo final
        timestamp_end = datetime.now()
        duration = timestamp_end - timestamp_start
        
        print(f"\n{'=' * 80}")
        print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"{'=' * 80}")
        print(f"\nInício:   {timestamp_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Término:  {timestamp_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duração:  {duration}")
        
        print(f"\n{'=' * 40}")
        print("COMPARAÇÃO DE MODELOS")
        print(f"{'=' * 40}")
        
        print("\nXGBoost Validation:")
        print(f"  Accuracy:  {classifier.results['xgboost']['val']['accuracy']:.4f}")
        print(f"  F1-Score:  {classifier.results['xgboost']['val']['f1']:.4f}")
        if classifier.results['xgboost']['val'].get('auc_roc'):
            print(f"  AUC-ROC:   {classifier.results['xgboost']['val']['auc_roc']:.4f}")
        
        print("\nRandom Forest Validation:")
        print(f"  Accuracy:  {classifier.results['random_forest']['val']['accuracy']:.4f}")
        print(f"  F1-Score:  {classifier.results['random_forest']['val']['f1']:.4f}")
        if classifier.results['random_forest']['val'].get('auc_roc'):
            print(f"  AUC-ROC:   {classifier.results['random_forest']['val']['auc_roc']:.4f}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
