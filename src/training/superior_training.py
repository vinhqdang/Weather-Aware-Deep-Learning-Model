"""
Superior Training Pipeline to Beat Tree-Based Models
This implements aggressive deep learning optimization to surpass Random Forest and XGBoost
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# XGBoost for comparison
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import sys
sys.path.append('../models')
from weather_aware_stgat import WeatherAwareSTGAT
from baseline_models import get_baseline_models, MLBaselines

class SuperiorTrainer:
    """Advanced trainer specifically designed to beat tree-based models"""
    
    def __init__(self, model, device='cuda', learning_rate=0.0003):
        self.model = model.to(device)
        self.device = device
        
        # Aggressive optimizer setup
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.005,  # Reduced for better learning
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # More aggressive learning rate schedule
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate * 3,  # Peak at 3x base LR
            epochs=100,
            steps_per_epoch=300,  # Approximate
            pct_start=0.3,  # Warmup for 30% of training
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=0.1)
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.val_r2_scores = []
        self.best_val_loss = float('inf')
        self.best_r2 = -float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def compute_superior_loss(self, predictions, targets):
        """Multi-component loss designed for superior performance"""
        # Combine MSE and Huber loss for robustness
        mse_loss = self.mse_loss(predictions, targets)
        huber_loss = self.huber_loss(predictions, targets)
        l1_loss = self.l1_loss(predictions, targets)
        
        # Weighted combination
        total_loss = 0.6 * mse_loss + 0.3 * huber_loss + 0.1 * l1_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'huber_loss': huber_loss.item(),
            'l1_loss': l1_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch):
        """Optimized training epoch"""
        self.model.train()
        total_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions, weather_logits, attention_weights = self.model(features)
            loss, loss_components = self.compute_superior_loss(predictions, targets)
            
            # Backward pass with gradient accumulation for stability
            self.optimizer.zero_grad()
            loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_losses.append(loss.item())
            
            # Update progress
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
        
        return np.mean(total_losses)
    
    def validate_epoch(self, val_loader):
        """Comprehensive validation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        val_losses = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions, _, _ = self.model(features)
                loss, _ = self.compute_superior_loss(predictions, targets)
                
                val_losses.append(loss.item())
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Calculate comprehensive metrics
        predictions_array = np.concatenate(all_predictions, axis=0)
        targets_array = np.concatenate(all_targets, axis=0)
        
        # Flatten for sklearn metrics
        pred_flat = predictions_array.flatten()
        target_flat = targets_array.flatten()
        
        mse = mean_squared_error(target_flat, pred_flat)
        mae = mean_absolute_error(target_flat, pred_flat)
        r2 = r2_score(target_flat, pred_flat)
        rmse = np.sqrt(mse)
        
        return np.mean(val_losses), {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def train(self, train_data, val_data, epochs=100, batch_size=256, patience=25):
        """Superior training to beat tree models"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print(f"🚀 SUPERIOR TRAINING to beat Random Forest and XGBoost!")
        print(f"Training samples: {len(X_train):,}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Target: Beat Random Forest (R² = 0.7782) and XGBoost")
        
        # Create optimized data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=4, pin_memory=True)
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Track progress
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_r2_scores.append(val_metrics['r2'])
            
            # Check for improvement
            if val_metrics['r2'] > self.best_r2:
                self.best_r2 = val_metrics['r2']
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                print(f"🎯 NEW BEST MODEL! R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
                
                # Check if we beat Random Forest
                if val_metrics['r2'] > 0.7782:
                    print(f"🔥 BEAT RANDOM FOREST! R² = {val_metrics['r2']:.4f} > 0.7782")
                
            else:
                self.patience_counter += 1
                print(f"⏳ Patience: {self.patience_counter}/{patience}")
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\n🏆 TRAINING COMPLETED!")
        print(f"Best R²: {self.best_r2:.4f}")
        print(f"Best RMSE: {np.sqrt(self.best_val_loss):.4f}")
        
        if self.best_r2 > 0.7782:
            print("✅ SUCCESSFULLY BEAT RANDOM FOREST!")
        else:
            print("❌ Need more optimization to beat Random Forest")
        
        return {
            'best_r2': self.best_r2,
            'best_rmse': np.sqrt(self.best_val_loss),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_r2_scores': self.val_r2_scores
        }

def train_superior_baselines(X_train, y_train, X_test, y_test):
    """Train optimized Random Forest and XGBoost for fair comparison"""
    
    print("🌳 Training optimized Random Forest and XGBoost baselines...")
    
    # Flatten data for sklearn - properly handle sequences
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # For multi-output regression, take the mean of the sequence as target
    if len(y_train.shape) > 2:
        y_train_flat = y_train.mean(axis=(1, 2))  # Average over sequence and features
        y_test_flat = y_test.mean(axis=(1, 2))
    elif len(y_train.shape) == 2:
        y_train_flat = y_train.mean(axis=1)  # Average over sequence
        y_test_flat = y_test.mean(axis=1)
    else:
        y_train_flat = y_train.flatten()
        y_test_flat = y_test.flatten()
    
    print(f"Flattened shapes: X_train {X_train_flat.shape}, y_train {y_train_flat.shape}")
    
    results = {}
    
    # Optimized Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,  # More trees
        max_depth=20,      # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_flat, y_train_flat)
    rf_pred = rf.predict(X_test_flat)
    
    results['random_forest'] = {
        'rmse': np.sqrt(mean_squared_error(y_test_flat, rf_pred)),
        'mae': mean_absolute_error(y_test_flat, rf_pred),
        'r2': r2_score(y_test_flat, rf_pred)
    }
    
    # Optimized XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_flat, y_train_flat)
    xgb_pred = xgb_model.predict(X_test_flat)
    
    results['xgboost'] = {
        'rmse': np.sqrt(mean_squared_error(y_test_flat, xgb_pred)),
        'mae': mean_absolute_error(y_test_flat, xgb_pred),
        'r2': r2_score(y_test_flat, xgb_pred)
    }
    
    print(f"Random Forest - R²: {results['random_forest']['r2']:.4f}, RMSE: {results['random_forest']['rmse']:.4f}")
    print(f"XGBoost - R²: {results['xgboost']['r2']:.4f}, RMSE: {results['xgboost']['rmse']:.4f}")
    
    return results

def create_optimized_weather_stgat(num_features, device='cuda'):
    """Create optimized Weather-Aware STGAT to beat tree models"""
    
    # Larger, more powerful architecture
    model = WeatherAwareSTGAT(
        num_features=num_features,
        weather_features=5,
        hidden_dim=256,      # Larger hidden dimension
        weather_dim=128,     # Larger weather embedding
        num_layers=6,        # Deeper network
        num_heads=16,        # More attention heads
        prediction_length=12,
        dropout=0.1,         # Lower dropout for more capacity
        num_nodes=1
    )
    
    # Apply weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model

def run_superior_training():
    """Main function to beat Random Forest and XGBoost"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 SUPERIOR TRAINING MODE - Target: Beat Random Forest & XGBoost")
    print(f"Device: {device}")
    
    # Load complete dataset
    print("📂 Loading complete dataset...")
    X_train = np.load('../../data/processed/X_train.npy')
    y_train = np.load('../../data/processed/y_train.npy')
    X_val = np.load('../../data/processed/X_val.npy')
    y_val = np.load('../../data/processed/y_val.npy')
    X_test = np.load('../../data/processed/X_test.npy')
    y_test = np.load('../../data/processed/y_test.npy')
    
    print(f"Training: {X_train.shape[0]:,} samples")
    print(f"Validation: {X_val.shape[0]:,} samples")
    print(f"Test: {X_test.shape[0]:,} samples")
    
    # Train baselines first
    baseline_results = train_superior_baselines(X_train, y_train, X_test, y_test)
    
    target_r2 = max(baseline_results['random_forest']['r2'], baseline_results['xgboost']['r2'])
    print(f"\n🎯 TARGET TO BEAT: R² = {target_r2:.4f}")
    
    # Create optimized model
    print("🏗️ Creating optimized Weather-Aware STGAT...")
    weather_model = create_optimized_weather_stgat(X_train.shape[-1], device)
    
    total_params = sum(p.numel() for p in weather_model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create superior trainer
    trainer = SuperiorTrainer(
        model=weather_model,
        device=device,
        learning_rate=0.0003
    )
    
    # Train to beat baselines
    print("🚀 Starting superior training...")
    training_results = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        patience=30
    )
    
    # Final evaluation on test set
    print("\n📊 FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    trainer.model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            predictions, _, _ = trainer.model(features)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate final metrics
    final_predictions = np.concatenate(all_predictions, axis=0).flatten()
    final_targets = np.concatenate(all_targets, axis=0).flatten()
    
    final_results = {
        'weather_aware_stgat': {
            'rmse': np.sqrt(mean_squared_error(final_targets, final_predictions)),
            'mae': mean_absolute_error(final_targets, final_predictions),
            'r2': r2_score(final_targets, final_predictions)
        }
    }
    
    # Combine all results
    all_results = {**baseline_results, **final_results}
    
    print("\n🏆 FINAL COMPARISON RESULTS")
    print("="*50)
    
    for model_name, metrics in all_results.items():
        print(f"{model_name.upper():<20}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Check if we won
    our_r2 = final_results['weather_aware_stgat']['r2']
    rf_r2 = baseline_results['random_forest']['r2']
    xgb_r2 = baseline_results['xgboost']['r2']
    
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"Weather-Aware STGAT: {our_r2:.4f}")
    print(f"Random Forest:       {rf_r2:.4f}")
    print(f"XGBoost:            {xgb_r2:.4f}")
    
    if our_r2 > rf_r2 and our_r2 > xgb_r2:
        print("🔥🔥🔥 SUCCESS! BEAT BOTH RANDOM FOREST AND XGBOOST! 🔥🔥🔥")
    elif our_r2 > max(rf_r2, xgb_r2):
        print("🎉 BEAT THE BEST BASELINE!")
    else:
        print("❌ Need more optimization...")
    
    # Save results
    os.makedirs('../../results/superior_training', exist_ok=True)
    
    complete_results = {
        'final_comparison': all_results,
        'training_history': training_results,
        'model_parameters': total_params,
        'target_beaten': our_r2 > max(rf_r2, xgb_r2),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('../../results/superior_training/superior_results.json', 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    return complete_results

if __name__ == "__main__":
    results = run_superior_training()