#!/usr/bin/env python3
"""
Complete training script for Alzheimer classification with MIRIAD dataset
- Cross-validation with patient-based splitting (no data leakage)
- NIfTI (.nii) file support with nibabel
- Binary classification: AD vs HC
- Complete MLflow logging with fold metrics

Improvements applied:
- CV by patient using StratifiedGroupKFold (fallback to GroupKFold)
- Augmentation order: PIL ops -> ToTensor -> Normalize
- safe_load_checkpoint fixed (no invalid kwargs) and map_location used
- avoid loading .nii for label extraction (use metadata in samples)
- robust ensure_3_channels
- TrainingLogger writes both to file and terminal
- removed duplicated methods in dataset class
- reproducible dataloader shuffling with generator and worker_init_fn
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import os
import sys
import numpy as np
import time
import atexit
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import nibabel as nib
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score, roc_auc_score
)

# Try to import StratifiedGroupKFold; fallback to GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGKF = True
except Exception:
    from sklearn.model_selection import GroupKFold
    _HAS_SGKF = False

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Path setup (optional)
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class ConvertToRGB:
    """A transform to ensure images are in RGB format."""
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img


def ensure_3_channels(tensor):
    """
    Ensures a tensor has 3 channels by repeating the first one if it only has 1.
    Robust for shapes: [C,H,W] or [B,C,H,W]
    """
    if tensor is None:
        return tensor
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        # [C, H, W] -> [3, H, W]
        return tensor.repeat(3, 1, 1)
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        # [B, 1, H, W] -> [B, 3, H, W]
        return tensor.repeat(1, 3, 1, 1)
    return tensor


class TrainingLogger:
    """Safe logger class that writes to file and also forwards to original terminal."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.__stdout__
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.file_handle = open(log_file, 'w', encoding='utf-8')
        self.closed = False

    def write(self, message):
        if self.closed:
            return
        try:
            self.file_handle.write(message)
            self.file_handle.flush()
        except Exception:
            pass
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except Exception:
            pass

    def flush(self):
        if self.closed:
            return
        try:
            self.file_handle.flush()
        except Exception:
            pass

    def close(self):
        if not self.closed:
            try:
                self.file_handle.flush()
                self.file_handle.close()
            except Exception:
                pass
            finally:
                self.closed = True


def setup_logging(model_name, timestamp):
    """Setup safe logging system and redirect stdout/stderr."""
    os.makedirs("./experiments/logs", exist_ok=True)
    log_file = f"./experiments/logs/train_miriad_cv_{model_name}_{timestamp}.log"

    logger = TrainingLogger(log_file)

    def cleanup_logger():
        if logger and not logger.closed:
            logger.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    atexit.register(cleanup_logger)
    # Redirect stdout/stderr to logger object
    sys.stdout = logger
    sys.stderr = logger

    print("=" * 80)
    print(f"MIRIAD CROSS-VALIDATION TRAINING LOG - {model_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Log file: {log_file}")
    print("=" * 80)
    print()

    return logger, log_file


class MIRIADDataset(Dataset):
    """Custom dataset for MIRIAD .nii files"""
    def __init__(self, data_dir, transform=None, slice_type='axial', slice_range=None):
        """
        Args:
            data_dir: Path to MIRIAD data directory
            transform: Transformations to apply
            slice_type: 'axial', 'coronal', 'sagittal'
            slice_range: tuple (start, end) for slice selection, None for middle slices
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.slice_type = slice_type
        self.slice_range = slice_range

        # Find all .nii files
        self.samples = []
        self.patients = {}
        self.class_to_idx = {'AD': 0, 'HC': 1}
        self.idx_to_class = {0: 'AD', 1: 'HC'}

        print(f"Scanning MIRIAD directory: {data_dir}")
        self._scan_directory()

        print(f"Found {len(self.samples)} samples from {len(self.patients)} patients")
        self._print_class_distribution()

    def _scan_directory(self):
        """Scan directory for MIRIAD files"""
        nii_files = list(self.data_dir.rglob("*.nii"))
        for nii_file in nii_files:
            filename = nii_file.stem
            parts = filename.split('_')
            # Expecting format: miriad_PPP_GR_G_VV_MR_S.nii (at least)
            if len(parts) >= 3 and parts[0] == 'miriad':
                # Be defensive when parsing
                patient_id = parts[1] if len(parts) > 1 else 'unknown'
                group = parts[2] if len(parts) > 2 else 'HC'
                gender = parts[3] if len(parts) > 3 else 'U'
                visit = parts[4] if len(parts) > 4 else '00'

                if group not in self.class_to_idx:
                    # Skip unknown groups
                    continue

                label = self.class_to_idx[group]
                sample_idx = len(self.samples)
                self.samples.append({
                    'file_path': nii_file,
                    'patient_id': patient_id,
                    'group': group,
                    'gender': gender,
                    'visit': visit,
                    'label': label
                })

                # Track patients
                if patient_id not in self.patients:
                    self.patients[patient_id] = {
                        'group': group,
                        'gender': gender,
                        'scans': []
                    }
                self.patients[patient_id]['scans'].append(sample_idx)

    def _print_class_distribution(self):
        """Print class distribution"""
        class_counts = defaultdict(int)
        patient_counts = defaultdict(int)

        for sample in self.samples:
            class_counts[sample['group']] += 1

        for patient_id, patient_info in self.patients.items():
            patient_counts[patient_info['group']] += 1

        print("Class distribution (scans):")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} scans")

        print("Class distribution (patients):")
        for class_name, count in patient_counts.items():
            print(f"  {class_name}: {count} patients")

    def _extract_slice(self, nii_data):
        """Extract 2D slice from 3D volume"""
        if self.slice_type == 'axial':
            slice_idx = nii_data.shape[2] // 2
            if self.slice_range:
                start, end = self.slice_range
                slice_idx = start + (end - start) // 2
            slice_2d = nii_data[:, :, slice_idx]

        elif self.slice_type == 'coronal':
            slice_idx = nii_data.shape[1] // 2
            if self.slice_range:
                start, end = self.slice_range
                slice_idx = start + (end - start) // 2
            slice_2d = nii_data[:, slice_idx, :]

        elif self.slice_type == 'sagittal':
            slice_idx = nii_data.shape[0] // 2
            if self.slice_range:
                start, end = self.slice_range
                slice_idx = start + (end - start) // 2
            slice_2d = nii_data[slice_idx, :, :]

        else:
            raise ValueError(f"Unknown slice_type: {self.slice_type}")

        return slice_2d

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            # Load NIfTI file
            nii = nib.load(sample['file_path'])
            nii_data = nii.get_fdata()

            # Extract 2D slice
            slice_2d = self._extract_slice(nii_data)

            # Normalize to 0-255 and convert to uint8
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            slice_2d = (slice_2d * 255).astype(np.uint8)

            # Convert to PIL Image (grayscale -> RGB by ConvertToRGB in transforms)
            image = Image.fromarray(slice_2d)

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            label = sample['label']
            return image, label

        except Exception as e:
            print(f"Error loading {sample['file_path']}: {e}")
            # Return a dummy RGB image (3 channels)
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, sample['label']

    def get_patient_labels(self):
        """Get labels for patient-based stratification"""
        patient_labels = []
        patient_ids = []
        for patient_id, patient_info in self.patients.items():
            label = self.class_to_idx[patient_info['group']]
            patient_labels.append(label)
            patient_ids.append(patient_id)
        return patient_ids, patient_labels

    def get_samples_for_patients(self, patient_ids):
        """Get sample indices for given patient IDs"""
        indices = []
        for patient_id in patient_ids:
            if patient_id in self.patients:
                indices.extend(self.patients[patient_id]['scans'])
        return indices


class MinMaxNormalize:
    """Min-Max normalization (if needed in pipeline)"""
    def __call__(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)


class EarlyStopping:
    """Early stopping"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def safe_load_checkpoint(checkpoint_path, map_location=None):
    """Load checkpoint safely with map_location support"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        print(f"Checkpoint loaded successfully: {checkpoint_path}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        raise e


def setup_mlflow():
    """Setup MLflow with proper error handling"""
    tracking_uri = "file:./experiments/mlruns"
    os.makedirs("./experiments/mlruns", exist_ok=True)
    os.makedirs("./experiments/models", exist_ok=True)
    os.makedirs("./experiments/plots", exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow configured: {tracking_uri}")
    return tracking_uri


def get_data_transforms():
    """Data transformations: apply PIL augmentations before ToTensor"""
    train_transform = transforms.Compose([
        ConvertToRGB(),  # ensure PIL RGB
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def calculate_complete_metrics(y_true, y_pred, y_proba=None, class_names=None):
    """Calculate complete metrics for binary classification"""
    if class_names is None:
        class_names = ['AD', 'HC']

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Sensitivity = Recall
    sensitivity_per_class = recall_per_class
    sensitivity_avg = np.mean(sensitivity_per_class)

    # Specificity, PPV, NPV per class
    specificity_per_class = []
    ppv_per_class = []
    npv_per_class = []

    for i in range(len(class_names)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        fn = np.sum(cm[i, :]) - cm[i, i]
        tp = cm[i, i]

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        specificity_per_class.append(specificity)
        ppv_per_class.append(ppv)
        npv_per_class.append(npv)

    # Averages
    specificity_avg = np.mean(specificity_per_class)
    ppv_avg = np.mean(ppv_per_class)
    npv_avg = np.mean(npv_per_class)

    # AUC-ROC for binary classification
    auc_roc = None
    if y_proba is not None and len(class_names) == 2:
        try:
            auc_roc = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            auc_roc = None

    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'mcc': mcc,
        'sensitivity_avg': sensitivity_avg,
        'specificity_avg': specificity_avg,
        'ppv_avg': ppv_avg,
        'npv_avg': npv_avg,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'sensitivity_per_class': sensitivity_per_class,
        'specificity_per_class': specificity_per_class,
        'ppv_per_class': ppv_per_class,
        'npv_per_class': npv_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

    return metrics


def create_model(model_config):
    """Factory for models - FIXED IMPORTS"""
    model_name = model_config['name']
    num_classes = 2  # Binary classification: AD vs HC
    dropout_rate = model_config.get('dropout_rate', 0.5)

    print(f"Creating model: {model_name} (Binary classification: AD vs HC)")

    # Make sure project src is in path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        if model_name == 'cnn_manual':
            from models.cnn_manual import CNNManual
            model = CNNManual(num_classes=num_classes)
            if not hasattr(model, 'dropout'):
                model.dropout = nn.Dropout(dropout_rate)
            return model

        elif model_name == 'vit_bt':
            from models.vit_bt import ViTBT
            return ViTBT(num_classes=num_classes)

        elif model_name == 'huggingface':
            from models.huggingface_model import HuggingFaceOASIS
            return HuggingFaceOASIS(num_classes=num_classes)

        elif model_name == 'resnet152':
            from models.pretrained_models import ResNet152OASIS
            return ResNet152OASIS(num_classes=num_classes)

        elif model_name in ['efficientnet_v2m', 'efficientnet']:
            from models.pretrained_models import EfficientNetV2M
            return EfficientNetV2M(num_classes=num_classes)

        else:
            raise ValueError(f"Model {model_name} not implemented")

    except ImportError as e:
        print(f"? ERRO ao importar modelo {model_name}: {e}")
        print(f"?? Verificar se existe: {src_dir}/models/{model_name}.py")
        print(f"?? Python path: {sys.path[:3]}...")
        raise ImportError(f"Cannot import model {model_name}. Check if models/{model_name}.py exists.") from e


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch_num, fold_num):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_proba = []

    print(f"  Training Fold {fold_num} - Epoch {epoch_num}...")

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Ensure 3 channels
        inputs = ensure_3_channels(inputs)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Predictions
        preds = outputs.argmax(dim=1)
        proba = torch.softmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_proba.extend(proba.cpu().detach().numpy())

        # Reduced batch logging for CV
        if batch_idx % 20 == 0:
            batch_acc = (preds == labels).float().mean().item()
            print(f"    Fold {fold_num} - Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}")

    epoch_loss = running_loss / max(1, len(data_loader.dataset))
    metrics = calculate_complete_metrics(y_true, y_pred, np.array(y_proba))

    return epoch_loss, metrics


def validate_one_epoch(model, data_loader, criterion, device, phase="Validation"):
    """Validate/Test one epoch"""
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_proba = []

    print(f"  Executing {phase.lower()}...")

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Ensure 3 channels
            inputs = ensure_3_channels(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Predictions
            preds = outputs.argmax(dim=1)
            proba = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(proba.cpu().detach().numpy())

    epoch_loss = running_loss / max(1, len(data_loader.dataset))
    metrics = calculate_complete_metrics(y_true, y_pred, np.array(y_proba))

    return epoch_loss, metrics


def log_fold_metrics(metrics, loss, fold_num, phase, step=None):
    """Log fold-specific metrics to MLflow"""
    mlflow_metrics = {
        f'{phase}_fold{fold_num}_loss': loss,
        f'{phase}_fold{fold_num}_accuracy': metrics['accuracy'],
        f'{phase}_fold{fold_num}_balanced_accuracy': metrics['balanced_accuracy'],
        f'{phase}_fold{fold_num}_f1_weighted': metrics['f1_weighted'],
        f'{phase}_fold{fold_num}_f1_macro': metrics['f1_macro'],
        f'{phase}_fold{fold_num}_sensitivity': metrics['sensitivity_avg'],
        f'{phase}_fold{fold_num}_specificity': metrics['specificity_avg'],
        f'{phase}_fold{fold_num}_ppv': metrics['ppv_avg'],
        f'{phase}_fold{fold_num}_npv': metrics['npv_avg'],
        f'{phase}_fold{fold_num}_mcc': metrics['mcc']
    }

    if metrics['auc_roc'] is not None:
        mlflow_metrics[f'{phase}_fold{fold_num}_auc_roc'] = metrics['auc_roc']

    if step is not None:
        mlflow.log_metrics(mlflow_metrics, step=step)
    else:
        mlflow.log_metrics(mlflow_metrics)


def print_metrics_summary(metrics, loss, phase, fold_num=None):
    """Print metrics summary"""
    fold_str = f" (Fold {fold_num})" if fold_num else ""
    print(f"  {phase}{fold_str}:")
    print(f"    Loss: {loss:.4f}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"    F1-weighted: {metrics['f1_weighted']:.4f}")
    print(f"    F1-macro: {metrics['f1_macro']:.4f}")
    print(f"    Sensitivity: {metrics['sensitivity_avg']:.4f}")
    print(f"    Specificity: {metrics['specificity_avg']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")


def calculate_cv_statistics(fold_results):
    """Calculate cross-validation statistics"""
    metrics_names = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro',
                     'sensitivity_avg', 'specificity_avg', 'ppv_avg', 'npv_avg', 'mcc', 'auc_roc']

    cv_stats = {}

    for metric in metrics_names:
        values = [fold[metric] for fold in fold_results if fold.get(metric) is not None]
        if values:
            cv_stats[f'{metric}_mean'] = np.mean(values)
            cv_stats[f'{metric}_std'] = np.std(values)
            cv_stats[f'{metric}_values'] = values
        else:
            cv_stats[f'{metric}_mean'] = 0.0
            cv_stats[f'{metric}_std'] = 0.0
            cv_stats[f'{metric}_values'] = []

    return cv_stats


def get_activation_function(model):
    """Detect activation function"""
    activations = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            activations.append('ReLU')
        elif isinstance(module, nn.Sigmoid):
            activations.append('Sigmoid')
        elif isinstance(module, nn.Tanh):
            activations.append('Tanh')
        elif isinstance(module, nn.LeakyReLU):
            activations.append('LeakyReLU')
        elif isinstance(module, nn.GELU):
            activations.append('GELU')
        elif isinstance(module, nn.ELU):
            activations.append('ELU')

    return list(set(activations)) if activations else ['Unknown']


def save_confusion_matrix(cm, model_name, timestamp, class_names, suffix=""):
    """Save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.title(f'Confusion Matrix - {model_name}{suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = f"./experiments/plots/confusion_matrix_miriad_{model_name}_{timestamp}{suffix}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    return cm_path


def split_patients_train_test(dataset, test_ratio=0.2, random_state=42):
    """Split patients into train and test sets"""
    patient_ids, patient_labels = dataset.get_patient_labels()

    from sklearn.model_selection import train_test_split

    train_patients, test_patients, _, _ = train_test_split(
        patient_ids, patient_labels,
        test_size=test_ratio,
        stratify=patient_labels,
        random_state=random_state
    )

    train_indices = dataset.get_samples_for_patients(train_patients)
    test_indices = dataset.get_samples_for_patients(test_patients)

    print(f"Patient-based split:")
    print(f"  Train patients: {len(train_patients)}, Train samples: {len(train_indices)}")
    print(f"  Test patients: {len(test_patients)}, Test samples: {len(test_indices)}")

    return train_indices, test_indices, train_patients, train_patients  # returning train_patients twice for compatibility


def main():
    """Main function with Cross-Validation for MIRIAD"""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Train model with Cross-Validation on MIRIAD')
        parser.add_argument('--data_dir', type=str, required=True, help='Path to MIRIAD data directory')
        parser.add_argument('--config', type=str, required=True, help='YAML config file')
        parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
        parser.add_argument('--slice_type', type=str, default='axial', choices=['axial', 'coronal', 'sagittal'])
        args = parser.parse_args()

        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Config {args.config} is empty!")

        model_name = config['model']['name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Setup logging
        logger, log_file = setup_logging(model_name, timestamp)

        total_start_time = time.time()
        print("Starting Cross-Validation training on MIRIAD dataset...")
        print(f"Data directory: {args.data_dir}")
        print(f"Config: {args.config}")
        print(f"Model: {model_name}")
        print(f"Number of folds: {args.n_folds}")
        print(f"Slice type: {args.slice_type}")
        print()

        # Reproducibility
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # MLflow setup
        try:
            setup_mlflow()
            mlflow.set_experiment("miriad_alzheimer_classification_cv")
        except Exception as e:
            print(f"MLflow setup error: {e}")
            raise

        run_name = f"MIRIAD_{model_name}_CV{args.n_folds}_{timestamp}"

        with mlflow.start_run(run_name=run_name) as run:
            print(f"MLflow Run: {run_name}")
            print(f"Run ID: {run.info.run_id}")

            # Load data transforms
            train_transform, val_transform = get_data_transforms()

            # Create full dataset (initially with val transform to speed scanning)
            full_dataset = MIRIADDataset(
                data_dir=args.data_dir,
                transform=val_transform,
                slice_type=args.slice_type
            )

            # Patient-based train/test split to avoid data leakage
            train_indices, test_indices, train_patients, _ = split_patients_train_test(
                full_dataset, test_ratio=0.2, random_state=seed
            )

            # Create train and test datasets (Subset points to full_dataset)
            train_dataset = Subset(full_dataset, train_indices)
            test_dataset = Subset(full_dataset, test_indices)

            # Get labels for stratification (from metadata, avoid heavy I/O)
            train_sample_labels = [full_dataset.samples[i]['label'] for i in train_indices]
            train_sample_patient_ids = [full_dataset.samples[i]['patient_id'] for i in train_indices]

            # Log parameters
            mlflow.log_params({
                'model_name': model_name,
                'run_name': run_name,
                'config_file': args.config,
                'log_file': log_file,
                'data_dir': args.data_dir,
                'slice_type': args.slice_type,
                'cv_folds': args.n_folds,
                'num_classes': 2,  # Binary classification
                'epochs': config['training']['epochs'],
                'learning_rate': config['training']['learning_rate'],
                'optimizer': config['training']['optimizer'],
                'batch_size': config['dataset']['batch_size'],
                'seed': seed,
                'device': str(device),
                'total_samples': len(full_dataset),
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
                'train_patients': len(set(train_sample_patient_ids))
            })

            print(f"Dataset loaded:")
            print(f"  Total samples: {len(full_dataset)}")
            print(f"  Train samples: {len(train_dataset)}")
            print(f"  Test samples: {len(test_dataset)}")

            # Choose CV splitter: prefer StratifiedGroupKFold if available
            if _HAS_SGKF:
                print("Using StratifiedGroupKFold for CV (stratified by label and grouped by patient).")
                splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
                split_iter = splitter.split(X=np.arange(len(train_indices)), y=train_sample_labels, groups=train_sample_patient_ids)
            else:
                print("StratifiedGroupKFold not available. Falling back to GroupKFold (grouped by patient, no stratify).")
                from sklearn.model_selection import GroupKFold
                splitter = GroupKFold(n_splits=args.n_folds)
                split_iter = splitter.split(X=np.arange(len(train_indices)), groups=train_sample_patient_ids)

            # Store results from each fold
            fold_results = []
            best_fold_f1 = 0.0
            best_model_path = None

            print(f"\n=== STARTING {args.n_folds}-FOLD CROSS-VALIDATION ===")

            # DataLoader options (reproducible)
            dataloader_batch_size = config['dataset']['batch_size']
            num_workers = config['dataset'].get('num_workers', 2)
            # generator for deterministic shuffle
            g = torch.Generator()
            try:
                g.manual_seed(seed)
            except Exception:
                pass

            def worker_init_fn(worker_id):
                np.random.seed(seed + worker_id)

            # Cross-validation loop
            for fold, (fold_train_idx, fold_val_idx) in enumerate(split_iter):
                print(f"\n{'='*20} FOLD {fold + 1}/{args.n_folds} {'='*20}")

                # Map back to actual dataset indices
                fold_train_indices = [train_indices[i] for i in fold_train_idx]
                fold_val_indices = [train_indices[i] for i in fold_val_idx]

                # Create fold datasets (Subset)
                fold_train_dataset = Subset(full_dataset, fold_train_indices)
                fold_val_dataset = Subset(full_dataset, fold_val_indices)

                # Apply transforms for training/validation
                fold_train_dataset.dataset.transform = train_transform
                fold_val_dataset.dataset.transform = val_transform

                # Create data loaders
                fold_train_loader = DataLoader(
                    fold_train_dataset,
                    batch_size=dataloader_batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    worker_init_fn=worker_init_fn,
                    generator=g
                )

                fold_val_loader = DataLoader(
                    fold_val_dataset,
                    batch_size=dataloader_batch_size,
                    shuffle=False,
                    num_workers=num_workers
                )

                print(f"Fold {fold + 1} - Train: {len(fold_train_dataset)}, Val: {len(fold_val_dataset)}")
                # Create fresh model for each fold
                model = create_model(config['model'])
                model = model.to(device)

                # Log model info (only once)
                if fold == 0:
                    activations = get_activation_function(model)
                    activation_str = ', '.join(activations)
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                    mlflow.log_params({
                        'activation_functions': activation_str,
                        'total_parameters': total_params,
                        'trainable_parameters': trainable_params
                    })

                    print(f"Model info - Activations: {activation_str}")
                    print(f"Parameters: {trainable_params:,} trainable")

                # Setup optimizer and criterion
                optimizer_name = config['training'].get('optimizer', 'adamax')
                lr = config['training']['learning_rate']

                if optimizer_name.lower() == 'adamax':
                    optimizer = optim.Adamax(model.parameters(), lr=lr)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                criterion = nn.CrossEntropyLoss()
                early_stopping = EarlyStopping(config['training'].get('early_stopping_patience', 10))

                # Training loop for this fold
                num_epochs = config['training']['epochs']
                best_fold_val_f1 = 0.0

                for epoch in range(num_epochs):
                    print(f"\n--- Fold {fold + 1} - Epoch {epoch + 1}/{num_epochs} ---")

                    # Training
                    train_loss, train_metrics = train_one_epoch(
                        model, fold_train_loader, optimizer, criterion, device, epoch + 1, fold + 1
                    )
                    print_metrics_summary(train_metrics, train_loss, "TRAIN", fold + 1)

                    # Validation
                    val_loss, val_metrics = validate_one_epoch(
                        model, fold_val_loader, criterion, device, "VALIDATION"
                    )
                    print_metrics_summary(val_metrics, val_loss, "VAL", fold + 1)

                    # Log fold metrics
                    log_fold_metrics(train_metrics, train_loss, fold + 1, "train")
                    log_fold_metrics(val_metrics, val_loss, fold + 1, "val")

                    # Track best model for this fold
                    if val_metrics['f1_weighted'] > best_fold_val_f1:
                        best_fold_val_f1 = val_metrics['f1_weighted']

                        # Save best model overall
                        if val_metrics['f1_weighted'] > best_fold_f1:
                            best_fold_f1 = val_metrics['f1_weighted']
                            best_model_path = f"./experiments/models/best_miriad_cv_{model_name}_{timestamp}.pth"

                            checkpoint = {
                                'fold': fold + 1,
                                'epoch': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_f1': best_fold_f1,
                                'model_name': model_name,
                                'dataset': 'MIRIAD',
                                'slice_type': args.slice_type
                            }

                            torch.save(checkpoint, best_model_path)
                            print(f"NEW BEST MODEL - Fold {fold + 1}, F1: {best_fold_f1:.4f}")

                    # Early stopping
                    if early_stopping(val_loss):
                        print(f"Early stopping at Fold {fold + 1}, Epoch {epoch + 1}")
                        break

                # Store fold results
                fold_results.append({
                    'fold': fold + 1,
                    'accuracy': val_metrics['accuracy'],
                    'balanced_accuracy': val_metrics['balanced_accuracy'],
                    'f1_weighted': val_metrics['f1_weighted'],
                    'f1_macro': val_metrics['f1_macro'],
                    'sensitivity_avg': val_metrics['sensitivity_avg'],
                    'specificity_avg': val_metrics['specificity_avg'],
                    'ppv_avg': val_metrics['ppv_avg'],
                    'npv_avg': val_metrics['npv_avg'],
                    'mcc': val_metrics['mcc'],
                    'auc_roc': val_metrics['auc_roc']
                })

                print(f"\nFold {fold + 1} completed - F1: {val_metrics['f1_weighted']:.4f}")

            # Calculate CV statistics
            cv_stats = calculate_cv_statistics(fold_results)

            print(f"\n{'='*60}")
            print("CROSS-VALIDATION RESULTS SUMMARY")
            print("="*60)

            for metric in ['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro',
                          'sensitivity_avg', 'specificity_avg', 'ppv_avg', 'npv_avg', 'mcc', 'auc_roc']:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in cv_stats:
                    mean_val = cv_stats[mean_key]
                    std_val = cv_stats[std_key]
                    print(f"{metric:20}: {mean_val:.4f} +- {std_val:.4f}")

                    # Log CV statistics to MLflow
                    mlflow.log_metrics({
                        f'cv_{metric}_mean': mean_val,
                        f'cv_{metric}_std': std_val
                    })

            # Final test evaluation with best model
            print(f"\n{'='*60}")
            print("FINAL TEST EVALUATION")
            print("="*60)

            if best_model_path and os.path.exists(best_model_path):
                # Load best model
                checkpoint = safe_load_checkpoint(best_model_path, map_location=device)
                model = create_model(config['model'])
                # Use strict=False as protective measure in case head differs
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model = model.to(device)
                print(f"Best model loaded from Fold {checkpoint.get('fold', 'N/A')}")

            # Ensure test_dataset uses validation transform
            test_dataset.dataset.transform = val_transform
            test_loader = DataLoader(
                test_dataset,
                batch_size=config['dataset']['batch_size'],
                shuffle=False,
                num_workers=config['dataset'].get('num_workers', 2)
            )

            test_loss, test_metrics = validate_one_epoch(
                model, test_loader, criterion, device, "FINAL TEST"
            )

            print_metrics_summary(test_metrics, test_loss, "FINAL TEST")

            # Log test metrics
            mlflow.log_metrics({
                "final_test_accuracy": test_metrics['accuracy'],
                "final_test_balanced_accuracy": test_metrics['balanced_accuracy'],
                "final_test_f1_weighted": test_metrics['f1_weighted'],
                "final_test_f1_macro": test_metrics['f1_macro'],
                "final_test_sensitivity": test_metrics['sensitivity_avg'],
                "final_test_specificity": test_metrics['specificity_avg'],
                "final_test_ppv": test_metrics['ppv_avg'],
                "final_test_npv": test_metrics['npv_avg'],
                "final_test_mcc": test_metrics['mcc']
            })

            if test_metrics['auc_roc'] is not None:
                mlflow.log_metric("final_test_auc_roc", test_metrics['auc_roc'])

            # Total time
            total_time = time.time() - total_start_time
            mlflow.log_metric("total_execution_time", total_time)

            # Save confusion matrix
            class_names = ['AD', 'HC']
            cm_path = save_confusion_matrix(
                test_metrics['confusion_matrix'],
                model_name, timestamp, class_names, "_final_test"
            )
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # Log best model
            if best_model_path:
                mlflow.log_artifact(best_model_path, artifact_path="models")

            # Build description for MLflow tag
            auc_roc_cv_str = f"{cv_stats['auc_roc_mean']:.4f} +- {cv_stats['auc_roc_std']:.4f}" if cv_stats.get('auc_roc_mean', 0) > 0 else 'N/A'
            auc_roc_test_str = f"{test_metrics['auc_roc']:.4f}" if test_metrics['auc_roc'] is not None else 'N/A'

            description = f"""=== MIRIAD CROSS-VALIDATION SUMMARY - {model_name} ===

CONFIGURATION:
- Dataset: MIRIAD
- Model: {model_name}
- Slice Type: {args.slice_type}
- CV Folds: {args.n_folds}
- Epochs per fold: {config['training']['epochs']}
- Optimizer: {optimizer_name}
- Learning Rate: {lr}
- Batch Size: {config['dataset']['batch_size']}
- Seed: {seed}
- Total Samples: {len(full_dataset)}
- Train Samples: {len(train_dataset)}
- Test Samples: {len(test_dataset)}

CROSS-VALIDATION RESULTS (Mean +- Std):
- Accuracy: {cv_stats['accuracy_mean']:.4f} +- {cv_stats['accuracy_std']:.4f}
- Balanced Accuracy: {cv_stats['balanced_accuracy_mean']:.4f} +- {cv_stats['balanced_accuracy_std']:.4f}
- F1-weighted: {cv_stats['f1_weighted_mean']:.4f} +- {cv_stats['f1_weighted_std']:.4f}
- F1-macro: {cv_stats['f1_macro_mean']:.4f} +- {cv_stats['f1_macro_std']:.4f}
- Sensitivity: {cv_stats['sensitivity_avg_mean']:.4f} +- {cv_stats['sensitivity_avg_std']:.4f}
- Specificity: {cv_stats['specificity_avg_mean']:.4f} +- {cv_stats['specificity_avg_std']:.4f}
- PPV: {cv_stats['ppv_avg_mean']:.4f} +- {cv_stats['ppv_avg_std']:.4f}
- NPV: {cv_stats['npv_avg_mean']:.4f} +- {cv_stats['npv_avg_std']:.4f}
- AUC-ROC: {auc_roc_cv_str}

FINAL TEST RESULTS (Holdout):
- Accuracy: {test_metrics['accuracy']:.4f}
- Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}
- F1-weighted: {test_metrics['f1_weighted']:.4f}
- F1-macro: {test_metrics['f1_macro']:.4f}
- Sensitivity: {test_metrics['sensitivity_avg']:.4f}
- Specificity: {test_metrics['specificity_avg']:.4f}
- PPV: {test_metrics['ppv_avg']:.4f}
- NPV: {test_metrics['npv_avg']:.4f}
- AUC-ROC: {auc_roc_test_str}

EXECUTION TIME: {str(timedelta(seconds=int(total_time)))}

CONFUSION MATRIX (Final Test):
{test_metrics['confusion_matrix']}

Classes: AD (Alzheimer's Disease), HC (Healthy Control)
"""

            mlflow.set_tag("mlflow.note.content", description)

            # Final summary output
            print(f"\n" + "=" * 80)
            print(f"MIRIAD CROSS-VALIDATION COMPLETED - {model_name}")
            print(f"=" * 80)
            print(f"CV F1-weighted: {cv_stats['f1_weighted_mean']:.4f} +- {cv_stats['f1_weighted_std']:.4f}")
            print(f"Final Test F1: {test_metrics['f1_weighted']:.4f}")
            print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Final Test AUC-ROC: {auc_roc_test_str}")
            print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
            print(f"MLflow Run: {run_name}")
            print("=" * 80)

    except Exception as e:
        # Make sure we log the full traceback
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Ensure logger is closed
        if 'logger' in locals():
            try:
                logger.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
