"""
Baseline Models for Comparison with Weather-Aware STGAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LSTMBaseline(nn.Module):
    """
    LSTM Baseline Model
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, prediction_length=12, dropout=0.1):
        super(LSTMBaseline, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_length)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Generate predictions
        predictions = self.output_layer(last_hidden)
        
        return predictions

class GRUBaseline(nn.Module):
    """
    GRU Baseline Model
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, prediction_length=12, dropout=0.1):
        super(GRUBaseline, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_length)
        )
        
    def forward(self, x):
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Use last hidden state for prediction
        last_hidden = gru_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Generate predictions
        predictions = self.output_layer(last_hidden)
        
        return predictions

class TransformerBaseline(nn.Module):
    """
    Transformer Baseline Model
    """
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, prediction_length=12, dropout=0.1):
        super(TransformerBaseline, self).__init__()
        
        self.d_model = d_model
        self.prediction_length = prediction_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, prediction_length)
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)  # [batch_size, d_model]
        
        # Generate predictions
        predictions = self.output_layer(pooled)
        
        return predictions

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class CNNBaseline(nn.Module):
    """
    1D CNN Baseline Model
    """
    
    def __init__(self, input_dim, prediction_length=12, dropout=0.1):
        super(CNNBaseline, self).__init__()
        
        self.prediction_length = prediction_length
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, prediction_length)
        )
        
    def forward(self, x):
        # Transpose for conv1d [batch_size, features, seq_len]
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Global average pooling
        x = self.global_pool(x)  # [batch_size, 256, 1]
        x = x.squeeze(-1)  # [batch_size, 256]
        
        # Generate predictions
        predictions = self.output_layer(x)
        
        return predictions

class MLBaselines:
    """
    Traditional Machine Learning Baselines
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        self.fitted_models = {}
        
    def prepare_ml_data(self, X, y):
        """
        Prepare data for ML models (flatten sequences)
        """
        # Flatten the sequence dimension
        X_flat = X.reshape(X.shape[0], -1)  # [batch_size, seq_len * features]
        y_flat = y.reshape(y.shape[0], -1)  # [batch_size, prediction_length]
        
        return X_flat, y_flat
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train all ML baseline models
        """
        print("Training ML baseline models...")
        
        # Prepare data
        X_train_flat, y_train_flat = self.prepare_ml_data(X_train, y_train)
        X_val_flat, y_val_flat = self.prepare_ml_data(X_val, y_val)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_flat, y_train_flat)
            self.fitted_models[name] = model
            
            # Validate
            val_pred = model.predict(X_val_flat)
            
            # Compute metrics
            mse = mean_squared_error(y_val_flat, val_pred)
            mae = mean_absolute_error(y_val_flat, val_pred)
            r2 = r2_score(y_val_flat, val_pred)
            
            results[name] = {
                'val_mse': mse,
                'val_mae': mae,
                'val_r2': r2
            }
            
            print(f"{name} - Val MSE: {mse:.4f}, Val MAE: {mae:.4f}, Val R²: {r2:.4f}")
        
        return results
    
    def predict(self, X_test):
        """
        Make predictions with all fitted models
        """
        X_test_flat, _ = self.prepare_ml_data(X_test, np.zeros((X_test.shape[0], 12)))
        
        predictions = {}
        for name, model in self.fitted_models.items():
            predictions[name] = model.predict(X_test_flat)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate all models on test data
        """
        X_test_flat, y_test_flat = self.prepare_ml_data(X_test, y_test)
        
        results = {}
        predictions = {}
        
        for name, model in self.fitted_models.items():
            # Predict
            y_pred = model.predict(X_test_flat)
            predictions[name] = y_pred
            
            # Compute metrics
            mse = mean_squared_error(y_test_flat, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_flat, y_pred)
            r2 = r2_score(y_test_flat, y_pred)
            
            # MAPE
            mape = np.mean(np.abs((y_test_flat - y_pred) / (y_test_flat + 1e-8))) * 100
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            }
            
            print(f"{name} Test Results:")
            print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}, MAPE: {mape:.2f}%")
            print(f"  R²: {r2:.4f}")
        
        return results, predictions

def get_baseline_models(input_dim, prediction_length=12):
    """
    Get all deep learning baseline models
    """
    models = {
        'lstm': LSTMBaseline(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            prediction_length=prediction_length,
            dropout=0.1
        ),
        'gru': GRUBaseline(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            prediction_length=prediction_length,
            dropout=0.1
        ),
        'transformer': TransformerBaseline(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=3,
            prediction_length=prediction_length,
            dropout=0.1
        ),
        'cnn': CNNBaseline(
            input_dim=input_dim,
            prediction_length=prediction_length,
            dropout=0.1
        )
    }
    
    return models

def test_baseline_models():
    """Test all baseline models"""
    
    # Test parameters
    batch_size = 16
    seq_len = 12
    input_dim = 26
    prediction_length = 12
    
    # Create test data
    X = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, prediction_length)
    
    # Get models
    models = get_baseline_models(input_dim, prediction_length)
    
    print("Testing baseline models...")
    
    for name, model in models.items():
        print(f"\nTesting {name}:")
        
        # Forward pass
        predictions = model(X)
        
        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test loss computation
        loss = F.mse_loss(predictions, y)
        print(f"  Test loss: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        
    print("\nAll baseline models tested successfully!")

if __name__ == "__main__":
    test_baseline_models()