"""
ULTIMATE XGBoost Annihilator Training Pipeline
100M+ Parameter Deep Learning Monster to COMPLETELY DESTROY XGBoost
Target: R² > 0.8760 (MUST BEAT XGBOOST OR DIE TRYING)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from ultimate_xgboost_destroyer import create_ultimate_destroyer

class UltimateAnnihilatorTrainer:
    """The most advanced trainer ever created - designed to OBLITERATE XGBoost"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # ULTIMATE OPTIMIZATION STRATEGY - Multi-optimizer approach
        self.main_optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0002,  # Conservative start for stability
            weight_decay=0.0005,  # Very light regularization for maximum capacity
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True  # Adaptive gradient for better convergence
        )
        
        # Secondary optimizer for fine-tuning
        self.fine_optimizer = optim.SGD(
            model.parameters(),
            lr=0.00005,
            momentum=0.9,
            weight_decay=0.0001,
            nesterov=True
        )
        
        # REVOLUTIONARY learning rate scheduling - Combination of strategies
        self.main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.main_optimizer,
            T_0=15,  # Restart every 15 epochs
            T_mult=2,  # Double restart period each time
            eta_min=1e-7
        )
        
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.main_optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # ULTIMATE loss functions arsenal
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=0.05)
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.val_r2_scores = []
        self.best_val_loss = float('inf')
        self.best_r2 = -float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # XGBoost annihilation tracking
        self.xgboost_target = 0.8760
        self.annihilated_xgboost = False
        self.annihilation_epoch = None
        
        # Advanced training techniques
        self.use_gradient_accumulation = True
        self.accumulation_steps = 4
        self.use_mixed_precision = True
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
    def compute_ultimate_loss(self, predictions, targets, weather_logits, uncertainty, attention_dict):
        """The most sophisticated loss function ever created"""
        
        # Primary prediction losses with adaptive weighting
        mse_loss = self.mse_loss(predictions, targets)
        l1_loss = self.l1_loss(predictions, targets)
        huber_loss = self.huber_loss(predictions, targets)
        smooth_l1_loss = self.smooth_l1_loss(predictions, targets)
        
        # Adaptive loss weighting based on current performance
        current_mse = mse_loss.item()
        if current_mse > 0.1:  # Early training - focus on MSE
            prediction_loss = (0.6 * mse_loss + 0.2 * huber_loss + 0.1 * l1_loss + 0.1 * smooth_l1_loss)
        elif current_mse > 0.05:  # Mid training - balance
            prediction_loss = (0.4 * mse_loss + 0.3 * huber_loss + 0.2 * l1_loss + 0.1 * smooth_l1_loss)
        else:  # Late training - focus on robustness
            prediction_loss = (0.3 * mse_loss + 0.4 * huber_loss + 0.2 * l1_loss + 0.1 * smooth_l1_loss)
        
        # Uncertainty-aware loss
        if uncertainty is not None:
            # Penalize high uncertainty for accurate predictions
            uncertainty_penalty = torch.mean(uncertainty * torch.abs(predictions - targets))
            # Encourage reasonable uncertainty levels
            uncertainty_reg = torch.mean(torch.abs(uncertainty - 0.1))  # Target uncertainty around 0.1
        else:
            uncertainty_penalty = torch.tensor(0.0, device=predictions.device)
            uncertainty_reg = torch.tensor(0.0, device=predictions.device)
        
        # Ultra-advanced weather classification loss
        if weather_logits is not None:
            batch_size = weather_logits.shape[0]
            # Sophisticated pseudo-labeling based on input patterns
            weather_targets = self._generate_weather_pseudo_labels(predictions, targets, batch_size)
            weather_loss = F.cross_entropy(weather_logits, weather_targets)
        else:
            weather_loss = torch.tensor(0.0, device=predictions.device)
        
        # Expert diversity and specialization
        expert_weights = attention_dict.get('expert_weights', None)
        importance_weights = attention_dict.get('importance_weights', None)
        
        if expert_weights is not None:
            # Maximize expert diversity (entropy)
            expert_entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=1)
            expert_diversity_loss = -expert_entropy.mean()  # Negative to maximize entropy
            
            # Encourage expert specialization
            if importance_weights is not None:
                specialization_loss = -torch.mean(torch.max(importance_weights, dim=1)[0])
            else:
                specialization_loss = torch.tensor(0.0, device=predictions.device)
        else:
            expert_diversity_loss = torch.tensor(0.0, device=predictions.device)
            specialization_loss = torch.tensor(0.0, device=predictions.device)
        
        # Temporal consistency regularization
        temporal_attention = attention_dict.get('temporal_attention', None)
        if temporal_attention is not None and temporal_attention.dim() >= 3:
            # Encourage smooth temporal attention transitions
            temporal_consistency = torch.mean(torch.abs(temporal_attention[..., 1:] - temporal_attention[..., :-1]))
        else:
            temporal_consistency = torch.tensor(0.0, device=predictions.device)
        
        # Scale importance regularization
        scale_importance = attention_dict.get('scale_importance', None)
        if scale_importance is not None:
            # Encourage balanced use of all time scales
            scale_balance = torch.var(scale_importance)  # Minimize variance for balance
        else:
            scale_balance = torch.tensor(0.0, device=predictions.device)
        
        # Graph attention regularization
        graph_attention = attention_dict.get('graph_attention', None)
        if graph_attention is not None:
            # Encourage focused but not overly sparse attention
            graph_sparsity = torch.mean(torch.abs(graph_attention))
        else:
            graph_sparsity = torch.tensor(0.0, device=predictions.device)
        
        # ULTIMATE loss combination with adaptive weighting
        total_loss = (
            prediction_loss + 
            0.1 * weather_loss +
            0.02 * uncertainty_penalty +
            0.01 * uncertainty_reg +
            0.03 * expert_diversity_loss +
            0.02 * specialization_loss +
            0.01 * temporal_consistency +
            0.01 * scale_balance +
            0.005 * graph_sparsity
        )
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'huber_loss': huber_loss.item(),
            'smooth_l1_loss': smooth_l1_loss.item(),
            'weather_loss': weather_loss.item(),
            'uncertainty_penalty': uncertainty_penalty.item(),
            'uncertainty_reg': uncertainty_reg.item(),
            'expert_diversity_loss': expert_diversity_loss.item(),
            'specialization_loss': specialization_loss.item(),
            'temporal_consistency': temporal_consistency.item(),
            'scale_balance': scale_balance.item(),
            'graph_sparsity': graph_sparsity.item()
        }
    
    def _generate_weather_pseudo_labels(self, predictions, targets, batch_size):
        """Generate sophisticated weather pseudo-labels"""
        # Create pseudo-labels based on prediction patterns
        error_magnitude = torch.abs(predictions - targets).mean(dim=1)
        
        # Classify based on error patterns (32 classes)
        weather_labels = []
        for error in error_magnitude:
            if error < 0.05:
                label = torch.randint(0, 8, (1,)).item()  # Clear weather types
            elif error < 0.1:
                label = torch.randint(8, 16, (1,)).item()  # Mild weather types
            elif error < 0.2:
                label = torch.randint(16, 24, (1,)).item()  # Moderate weather types
            else:
                label = torch.randint(24, 32, (1,)).item()  # Severe weather types
            weather_labels.append(label)
        
        return torch.tensor(weather_labels, device=predictions.device)
    
    def train_epoch(self, train_loader, epoch):
        """ULTIMATE training epoch with all advanced techniques"""
        self.model.train()
        total_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} - ANNIHILATING XGBoost')
        
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Mixed precision training
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions, weather_logits, uncertainty, attention_dict = self.model(features)
                    loss, loss_components = self.compute_ultimate_loss(
                        predictions, targets, weather_logits, uncertainty, attention_dict
                    )
                    loss = loss / self.accumulation_steps  # Scale for gradient accumulation
            else:
                predictions, weather_logits, uncertainty, attention_dict = self.model(features)
                loss, loss_components = self.compute_ultimate_loss(
                    predictions, targets, weather_logits, uncertainty, attention_dict
                )
            
            # Gradient accumulation
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.main_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.scaler.step(self.main_optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.main_optimizer.step()
                
                self.main_optimizer.zero_grad()
            
            total_losses.append(loss.item() * self.accumulation_steps)
            
            # Update progress
            if batch_idx % 25 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                    'lr': f'{self.main_scheduler.get_last_lr()[0]:.6f}',
                    'target': f'>{self.xgboost_target:.4f}'
                })
        
        # Step scheduler
        self.main_scheduler.step()
        
        return np.mean(total_losses)
    
    def validate_epoch(self, val_loader):
        """Comprehensive validation with XGBoost annihilation detection"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        val_losses = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions, weather_logits, uncertainty, attention_dict = self.model(features)
                        loss, _ = self.compute_ultimate_loss(
                            predictions, targets, weather_logits, uncertainty, attention_dict
                        )
                else:
                    predictions, weather_logits, uncertainty, attention_dict = self.model(features)
                    loss, _ = self.compute_ultimate_loss(
                        predictions, targets, weather_logits, uncertainty, attention_dict
                    )
                
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
    
    def train(self, train_data, val_data, epochs=200, batch_size=64, patience=75):
        """ULTIMATE training to ANNIHILATE XGBoost once and for all"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print(f"🔥🔥🔥 ULTIMATE XGBOOST ANNIHILATOR TRAINING 🔥🔥🔥")
        print(f"Training samples: {len(X_train):,}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🎯 ULTIMATE TARGET: ANNIHILATE XGBoost R² = {self.xgboost_target:.4f}")
        print(f"🚀 ULTIMATE Architecture: 100M+ Parameters of Pure Intelligence")
        print(f"🔥 Mixed Precision: {self.use_mixed_precision}")
        print(f"🔥 Gradient Accumulation: {self.accumulation_steps} steps")
        
        # Create optimized data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=6, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=6, pin_memory=True)
        
        # Training loop
        for epoch in range(epochs):
            print(f"\\n{'='*100}")
            print(f"🔥 EPOCH {epoch + 1}/{epochs} - ANNIHILATING XGBOOST 🔥")
            print(f"Learning Rate: {self.main_scheduler.get_last_lr()[0]:.6f}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Step plateau scheduler
            self.plateau_scheduler.step(val_loss)
            
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
                
                # Check if we ANNIHILATED XGBoost
                if val_metrics['r2'] > self.xgboost_target:
                    if not self.annihilated_xgboost:
                        self.annihilation_epoch = epoch + 1
                        self.annihilated_xgboost = True
                    improvement = ((val_metrics['r2'] - self.xgboost_target) / self.xgboost_target) * 100
                    print(f"🔥🔥🔥 XGBOOST ANNIHILATED! R² = {val_metrics['r2']:.4f} > {self.xgboost_target:.4f} 🔥🔥🔥")
                    print(f"🚀 IMPROVEMENT: +{improvement:.2f}% over XGBoost!")
                    print(f"🏆 ANNIHILATION ACHIEVED AT EPOCH {self.annihilation_epoch}!")
                else:
                    gap = self.xgboost_target - val_metrics['r2']
                    progress = (val_metrics['r2'] / self.xgboost_target) * 100
                    print(f"⚡ Progress: {progress:.1f}% toward XGBoost Annihilation, Gap: {gap:.4f}")
                
            else:
                self.patience_counter += 1
                print(f"⏳ Patience: {self.patience_counter}/{patience}")
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            
            # Early stopping with extended patience after annihilation
            effective_patience = patience + 25 if self.annihilated_xgboost else patience
            if self.patience_counter >= effective_patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\\n🏆 ULTIMATE TRAINING COMPLETED! 🏆")
        print(f"Best R²: {self.best_r2:.4f}")
        print(f"Best RMSE: {np.sqrt(self.best_val_loss):.4f}")
        
        if self.best_r2 > self.xgboost_target:
            improvement = ((self.best_r2 - self.xgboost_target) / self.xgboost_target) * 100
            print("✅ 🔥🔥🔥 XGBOOST SUCCESSFULLY ANNIHILATED! 🔥🔥🔥 ✅")
            print(f"🚀 FINAL IMPROVEMENT: +{improvement:.2f}% over XGBoost!")
            print(f"🏆 ANNIHILATION EPOCH: {self.annihilation_epoch}")
        else:
            gap = self.xgboost_target - self.best_r2
            print(f"❌ Still {gap:.4f} R² points from total annihilation...")
            print("🔄 Need even more ULTIMATE optimization...")
        
        return {
            'best_r2': self.best_r2,
            'best_rmse': np.sqrt(self.best_val_loss),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_r2_scores': self.val_r2_scores,
            'annihilated_xgboost': self.annihilated_xgboost,
            'annihilation_epoch': self.annihilation_epoch
        }

def train_ultimate_xgboost(X_train, y_train, X_test, y_test):
    """Train the most optimized XGBoost possible"""
    
    print("🌳 Training ULTIMATE XGBoost baseline...")
    
    # Flatten data for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Handle target flattening
    if len(y_train.shape) > 2:
        y_train_flat = y_train.mean(axis=(1, 2))
        y_test_flat = y_test.mean(axis=(1, 2))
    elif len(y_train.shape) == 2:
        y_train_flat = y_train.mean(axis=1)
        y_test_flat = y_test.mean(axis=1)
    else:
        y_train_flat = y_train.flatten()
        y_test_flat = y_test.flatten()
    
    # ULTIMATE XGBoost configuration
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,  # More trees
        max_depth=12,      # Deeper trees
        learning_rate=0.05,  # Lower learning rate for better precision
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.85,
        colsample_bynode=0.85,
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.05,     # L1 regularization
        reg_lambda=0.05,    # L2 regularization
        min_child_weight=3,
        gamma=0.1,
        max_delta_step=1
    )
    
    xgb_model.fit(X_train_flat, y_train_flat)
    xgb_pred = xgb_model.predict(X_test_flat)
    
    xgb_r2 = r2_score(y_test_flat, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test_flat, xgb_pred))
    
    print(f"ULTIMATE XGBoost Performance - R²: {xgb_r2:.4f}, RMSE: {xgb_rmse:.4f}")
    
    return {
        'r2': xgb_r2,
        'rmse': xgb_rmse,
        'mae': mean_absolute_error(y_test_flat, xgb_pred)
    }

def run_ultimate_annihilation():
    """Main function to ANNIHILATE XGBoost once and for all"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥🔥🔥 ULTIMATE XGBOOST ANNIHILATOR ACTIVATED 🔥🔥🔥")
    print(f"Device: {device}")
    
    # Load complete dataset
    print("📂 Loading COMPLETE dataset for ULTIMATE ANNIHILATION...")
    X_train = np.load('../../data/processed/X_train.npy')
    y_train = np.load('../../data/processed/y_train.npy')
    X_val = np.load('../../data/processed/X_val.npy')
    y_val = np.load('../../data/processed/y_val.npy')
    X_test = np.load('../../data/processed/X_test.npy')
    y_test = np.load('../../data/processed/y_test.npy')
    
    print(f"Training: {X_train.shape[0]:,} samples")
    print(f"Validation: {X_val.shape[0]:,} samples")
    print(f"Test: {X_test.shape[0]:,} samples")
    
    # Train ULTIMATE XGBoost baseline
    xgboost_results = train_ultimate_xgboost(X_train, y_train, X_test, y_test)
    xgboost_target = xgboost_results['r2']
    
    print(f"\\n🎯 ULTIMATE XGBOOST TARGET TO ANNIHILATE: R² = {xgboost_target:.4f}")
    
    # Create ULTIMATE destroyer
    print("🏗️ Creating ULTIMATE XGBoost Destroyer...")
    ultimate_model = create_ultimate_destroyer(X_train.shape[-1], device)
    
    total_params = sum(p.numel() for p in ultimate_model.parameters())
    print(f"ULTIMATE Model Parameters: {total_params:,}")
    
    # Create ULTIMATE annihilator trainer
    annihilator_trainer = UltimateAnnihilatorTrainer(
        model=ultimate_model,
        device=device
    )
    annihilator_trainer.xgboost_target = xgboost_target
    
    # ANNIHILATE XGBoost
    print("🚀 DEPLOYING ULTIMATE XGBOOST ANNIHILATOR...")
    training_results = annihilator_trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=200,
        batch_size=32,  # Smaller batch for 100M+ parameter model
        patience=75
    )
    
    # Final evaluation on test set
    print("\\n📊 FINAL ANNIHILATION ASSESSMENT")
    print("="*100)
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    annihilator_trainer.model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            if annihilator_trainer.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions, _, _, _ = annihilator_trainer.model(features)
            else:
                predictions, _, _, _ = annihilator_trainer.model(features)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate final metrics
    final_predictions = np.concatenate(all_predictions, axis=0).flatten()
    final_targets = np.concatenate(all_targets, axis=0).flatten()
    
    final_results = {
        'ultimate_xgboost_destroyer': {
            'rmse': np.sqrt(mean_squared_error(final_targets, final_predictions)),
            'mae': mean_absolute_error(final_targets, final_predictions),
            'r2': r2_score(final_targets, final_predictions)
        },
        'xgboost': xgboost_results
    }
    
    print("🏆 FINAL ANNIHILATION RESULTS")
    print("="*100)
    
    our_r2 = final_results['ultimate_xgboost_destroyer']['r2']
    xgb_r2 = final_results['xgboost']['r2']
    
    print(f"ULTIMATE XGBoost Destroyer: R² = {our_r2:.4f}")
    print(f"XGBoost (TARGET):           R² = {xgb_r2:.4f}")
    
    if our_r2 > xgb_r2:
        improvement = ((our_r2 - xgb_r2) / xgb_r2) * 100
        print(f"\\n🔥🔥🔥 XGBOOST COMPLETELY ANNIHILATED! 🔥🔥🔥")
        print(f"🚀 ULTIMATE IMPROVEMENT: +{improvement:.2f}% over XGBoost!")
        print("✅ ULTIMATE MISSION ACCOMPLISHED!")
        print(f"🏆 ANNIHILATION EPOCH: {training_results.get('annihilation_epoch', 'N/A')}")
    else:
        gap = xgb_r2 - our_r2
        print(f"\\n❌ Still {gap:.4f} R² points from total annihilation")
        print("🔄 Need even more ULTIMATE optimization...")
    
    # Save results
    os.makedirs('../../results/ultimate_annihilation', exist_ok=True)
    
    complete_results = {
        'final_comparison': final_results,
        'training_history': training_results,
        'model_parameters': total_params,
        'xgboost_annihilated': our_r2 > xgb_r2,
        'improvement_over_xgboost': ((our_r2 - xgb_r2) / xgb_r2) * 100 if xgb_r2 > 0 else 0,
        'annihilation_epoch': training_results.get('annihilation_epoch'),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('../../results/ultimate_annihilation/annihilation_results.json', 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    return complete_results

if __name__ == "__main__":
    results = run_ultimate_annihilation()