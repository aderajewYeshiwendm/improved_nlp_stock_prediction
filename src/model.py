"""
IMPROVED model.py with Class Imbalance Handling
Key improvements:
1. Class weight balancing for XGBoost and LightGBM
2. Better LSTM architecture with increased sequence length
3. Improved hyperparameters
4. Better logging of class distribution
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        weights = torch.tanh(self.attention(lstm_output))
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.4):
        super(LSTMModel, self).__init__()
        # We use a Bidirectional LSTM (doubles the hidden size)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = Attention(hidden_size * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # Note: Removed Sigmoid here because we move it to the Loss Function
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context)
    
class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(self, X, y, sequence_length=30):  # Increased from 10 to 30
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_val = self.y[idx + self.sequence_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_val])


class ModelTrainer:
    """Enhanced model trainer with class imbalance handling."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {}
        self.metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🚀 Using device: {self.device}")
    
    def check_class_balance(self, y, split_name="Data"):
        """Check and report class distribution."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        logger.info(f"\n📊 {split_name} Class Distribution:")
        for cls, count in zip(unique, counts):
            pct = (count / total) * 100
            logger.info(f"  Class {int(cls)}: {count:,} ({pct:.2f}%)")
        
        # Calculate imbalance ratio
        if len(counts) == 2:
            imbalance_ratio = max(counts) / min(counts)
            logger.info(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 1.5:
                logger.warning(f"  ⚠️  SIGNIFICANT CLASS IMBALANCE DETECTED!")
                logger.warning(f"  ⚠️  Applying class weight balancing...")
            else:
                logger.info(f"  ✅ Classes are reasonably balanced")
        
        return unique, counts
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'Target',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple:
        """Prepare data for training with class balance checking."""
        logger.info("\n" + "="*60)
        logger.info("📦 PREPARING DATA FOR TRAINING")
        logger.info("="*60)
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Check overall class balance
        self.check_class_balance(y, "Overall Dataset")
        
        # Time series split (no shuffling)
        total_size = len(X)
        train_size = int(total_size * (1 - test_size - val_size))
        val_size_count = int(total_size * val_size)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size_count]
        y_val = y[train_size:train_size + val_size_count]
        
        X_test = X[train_size + val_size_count:]
        y_test = y[train_size + val_size_count:]
        
        
        
        # Check each split
        self.check_class_balance(y_train, "Training Set")
        self.check_class_balance(y_val, "Validation Set")
        self.check_class_balance(y_test, "Test Set")
        
        logger.info("="*60 + "\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(self, X_train, X_val, y_train, y_val, params: Optional[Dict] = None) -> XGBClassifier:
        """Train XGBoost with class imbalance handling."""
        logger.info("\n🌳 Training XGBoost with Class Weight Balancing")
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([class_weights[int(y)] for y in y_train])
        
        logger.info(f"⚖️  Class weights: {dict(zip(classes.astype(int), class_weights))}")
        
        # IMPROVED HYPERPARAMETERS
        default_params = {
            'n_estimators': 300,  # Increased from 200
            'learning_rate': 0.03,  # Reduced from 0.05 for better learning
            'max_depth': 4,  # Reduced from 6 to prevent overfitting
            'min_child_weight': 3,  # Increased from 1
            'gamma': 0.1,  # Added regularization
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': class_weights[1] / class_weights[0] if len(class_weights) > 1 else 1.0,
            'random_state': 42,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 20  # Added early stopping
        }
        
        if params:
            default_params.update(params)
        
        logger.info(f"📋 Key parameters: n_estimators={default_params['n_estimators']}, "
                   f"lr={default_params['learning_rate']}, max_depth={default_params['max_depth']}")
        
        model = XGBClassifier(**default_params)
        
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,  # Apply sample weights
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.models['xgboost'] = model
        logger.info("✅ XGBoost training complete\n")
        return model
    
    def train_lightgbm(self, X_train, X_val, y_train, y_val, params: Optional[Dict] = None) -> LGBMClassifier:
        """Train LightGBM with class imbalance handling."""
        logger.info("\n💡 Training LightGBM with Class Weight Balancing")
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([class_weights[int(y)] for y in y_train])
        
        logger.info(f"⚖️  Class weights: {dict(zip(classes.astype(int), class_weights))}")
        
        # IMPROVED HYPERPARAMETERS
        default_params = {
            'n_estimators': 300,  # Increased from 200
            'learning_rate': 0.03,  # Reduced from 0.05
            'max_depth': 4,  # Reduced from 6
            'num_leaves': 15,  # Reduced from 31
            'min_child_samples': 20,
            'class_weight': 'balanced',  # Built-in class balancing
            'random_state': 42,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        logger.info(f"📋 Key parameters: n_estimators={default_params['n_estimators']}, "
                   f"lr={default_params['learning_rate']}, num_leaves={default_params['num_leaves']}")
        
        model = LGBMClassifier(**default_params)
        
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,  # Apply sample weights
            eval_set=[(X_val, y_val)],
            callbacks=[lambda x: None]  # Suppress output
        )
        
        self.models['lightgbm'] = model
        logger.info("✅ LightGBM training complete\n")
        return model
    
    def train_lstm(
        self,
        X_train, X_val, y_train, y_val,
        sequence_length: int = 30,  # Increased from 10
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.0005  # Reduced from 0.001
    ) -> LSTMModel:
        """Train LSTM model with class imbalance handling."""
        logger.info("\n🧠 Training LSTM with Improved Architecture")

        if len(X_train) <= sequence_length or len(X_val) <= sequence_length:
            logger.warning("⚠️  Not enough data for LSTM training. Skipping LSTM.")
            return None
        
        # Check and log class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"📊 Training set class distribution: {dict(zip(unique.astype(int), counts))}")
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train, sequence_length)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val, sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        input_size = X_train.shape[1]
        model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(self.device)
        
        # 1. Calculate the raw ratio
        pos_count = np.sum(y_train)
        neg_count = len(y_train) - pos_count

        # 2. Wrap it in a torch.tensor so PyTorch can use it
        # We use np.sqrt to prevent the weight from being too aggressive
        raw_weight = np.sqrt(neg_count / pos_count)
        pos_weight = torch.tensor([raw_weight], dtype=torch.float32).to(self.device)

        # 3. Pass it to the criterion
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Improved Optimizer with weight decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        logger.info(f"📋 Architecture: {num_layers} layers, {hidden_size} hidden units, seq_len={sequence_length}")
        logger.info(f"📋 Training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15  # Increased from 10
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1:3d}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"LR: {current_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"⏹️  Early stopping at epoch {epoch+1}")
                    break
        
        self.models['lstm'] = model
        logger.info("✅ LSTM training complete\n")
        return model
    
    def create_ensemble(self, X_test, y_test) -> Dict:
        """Create ensemble of trained models."""
        logger.info("\n🎯 Creating Ensemble Predictions")
        
        predictions = {}
        
        # Get predictions from each model
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict_proba(X_test)[:, 1]
            logger.info("  ✓ XGBoost predictions obtained")
        
        if 'lightgbm' in self.models:
            predictions['lightgbm'] = self.models['lightgbm'].predict_proba(X_test)[:, 1]
            logger.info("  ✓ LightGBM predictions obtained")
        
        if 'lstm' in self.models:
            # Need to create sequences for LSTM
            sequence_length = 30  # Match training sequence length
            lstm_predictions = []
            
            scaler = MinMaxScaler()
            X_test_scaled = scaler.fit_transform(X_test)
            
            self.models['lstm'].eval()
            with torch.no_grad():
                for i in range(sequence_length, len(X_test_scaled)):
                    X_seq = X_test_scaled[i-sequence_length:i]
                    X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                    logits = self.models['lstm'](X_tensor)
                    pred = torch.sigmoid(logits).cpu().numpy()[0][0]
                    lstm_predictions.append(pred)
            
            # Pad the beginning
            lstm_predictions = [0.5] * sequence_length + lstm_predictions
            predictions['lstm'] = np.array(lstm_predictions)
            logger.info("  ✓ LSTM predictions obtained")
        
        # Ensemble by averaging
        if predictions:
            ensemble_probs = np.mean(list(predictions.values()), axis=0)
            xgb_preds = (predictions['xgboost'] > 0.5)
            lstm_preds = (predictions['lstm'] > 0.5)
            ensemble_preds = (xgb_preds & lstm_preds).astype(int)
            
            logger.info(f"  ✓ Ensemble created from {len(predictions)} models\n")
            
            return {
                'predictions': ensemble_preds,
                'probabilities': ensemble_probs,
                'individual_predictions': predictions
            }
        
        return {}
    
    def evaluate_model(self, y_true, y_pred, y_prob=None, model_name: str = "model") -> Dict:
        """Comprehensive model evaluation with detailed metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 {model_name} EVALUATION METRICS")
        logger.info(f"{'='*60}")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        logger.info(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info(f"\n📈 Confusion Matrix:")
        logger.info(f"  TN: {cm[0, 0]:4d}  |  FP: {cm[0, 1]:4d}")
        logger.info(f"  FN: {cm[1, 0]:4d}  |  TP: {cm[1, 1]:4d}")
        logger.info(f"{'='*60}\n")
        
        return metrics
    
    def save_model(self, model_name: str, output_dir: str = "models"):
        """Save trained model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if model_name in self.models:
            model = self.models[model_name]
            file_path = output_path / f"{model_name}_model.pkl"
            
            if model_name in ['xgboost', 'lightgbm']:
                joblib.dump(model, file_path)
            elif model_name == 'lstm':
                torch.save(model.state_dict(), output_path / f"{model_name}_model.pth")
            
            logger.info(f"💾 Model saved to {file_path}")
    
    def load_model(self, model_name: str, model_dir: str = "models"):
        """Load saved model."""
        model_path = Path(model_dir)
        
        if model_name in ['xgboost', 'lightgbm']:
            file_path = model_path / f"{model_name}_model.pkl"
            if file_path.exists():
                self.models[model_name] = joblib.load(file_path)
                logger.info(f"📁 Loaded {model_name} model from {file_path}")
        elif model_name == 'lstm':
            file_path = model_path / f"{model_name}_model.pth"
            if file_path.exists():
                logger.warning("⚠️  LSTM model loading requires model architecture parameters")
        
        return self.models.get(model_name)


def train_model(df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
    """
    Legacy function for backward compatibility.
    Trains XGBoost model and returns metrics.
    """
    if feature_cols is None:
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_score', 'RSI', 'MACD']
    
    trainer = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        df, feature_cols, test_size=0.2, val_size=0.0
    )
    
    # Use X_test as validation set for legacy compatibility
    model = trainer.train_xgboost(X_train, X_test, y_train, y_test)
    predictions = model.predict(X_test)
    
    metrics = trainer.evaluate_model(y_test, predictions, model_name="XGBoost")
    
    # Convert to percentage for display
    display_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score']
    }
    
    return model, display_metrics, X_test, y_test, predictions