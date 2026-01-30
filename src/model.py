
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import joblib
from pathlib import Path


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(self, X, y, sequence_length=30):
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
    """Enhanced model trainer with multiple algorithms."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {}
        self.metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'Target',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple:
        """Prepare data for training."""
        logger.info("Preparing data for training")
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Time series split (no shuffling)
        total_size = len(X)
        train_size = int(total_size * (1 - test_size - val_size))
        val_size = int(total_size * val_size)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(self, X_train, X_val, y_train, y_val, params: Optional[Dict] = None) -> XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost model")
        
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        if params:
            default_params.update(params)
        
        model = XGBClassifier(**default_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, X_val, y_train, y_val, params: Optional[Dict] = None) -> LGBMClassifier:
        """Train LightGBM model."""
        logger.info("Training LightGBM model")
        
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        model = LGBMClassifier(**default_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[logger.info]
        )
        
        self.models['lightgbm'] = model
        return model
    
    def train_lstm(
        self,
        X_train, X_val, y_train, y_val,
        sequence_length: int = 30,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ) -> LSTMModel:
        """Train LSTM model."""
        logger.info("Training LSTM model")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        input_size = X_train.shape[1]
        model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
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
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.models['lstm'] = model
        return model
    
    def create_ensemble(self, X_test, y_test) -> Dict:
        """Create ensemble of trained models."""
        logger.info("Creating ensemble predictions")
        
        predictions = {}
        
        # Get predictions from each model
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict_proba(X_test)[:, 1]
        
        if 'lightgbm' in self.models:
            predictions['lightgbm'] = self.models['lightgbm'].predict_proba(X_test)[:, 1]
        
        if 'lstm' in self.models:
            # Need to create sequences for LSTM
            sequence_length = 30
            lstm_predictions = []
            
            self.models['lstm'].eval()
            with torch.no_grad():
                for i in range(sequence_length, len(X_test)):
                    X_seq = X_test[i-sequence_length:i]
                    X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                    pred = self.models['lstm'](X_tensor).cpu().numpy()[0][0]
                    lstm_predictions.append(pred)
            
            # Pad the beginning
            lstm_predictions = [0.5] * sequence_length + lstm_predictions
            predictions['lstm'] = np.array(lstm_predictions)
        
        # Ensemble by averaging
        if predictions:
            ensemble_probs = np.mean(list(predictions.values()), axis=0)
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            
            return {
                'predictions': ensemble_preds,
                'probabilities': ensemble_probs,
                'individual_predictions': predictions
            }
        
        return {}
    
    def evaluate_model(self, y_true, y_pred, y_prob=None, model_name: str = "model") -> Dict:
        """Evaluate model performance."""
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
        
        logger.info(f"{model_name} Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
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
            
            logger.info(f"Model saved to {file_path}")
    
    def load_model(self, model_name: str, model_dir: str = "models"):
        """Load saved model."""
        model_path = Path(model_dir)
        
        if model_name in ['xgboost', 'lightgbm']:
            file_path = model_path / f"{model_name}_model.pkl"
            if file_path.exists():
                self.models[model_name] = joblib.load(file_path)
                logger.info(f"Loaded {model_name} model from {file_path}")
        elif model_name == 'lstm':
            file_path = model_path / f"{model_name}_model.pth"
            if file_path.exists():
                # Would need to recreate architecture
                logger.warning("LSTM model loading requires model architecture parameters")
        
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
