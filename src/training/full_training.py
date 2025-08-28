"""
Full-Scale Training and Analysis for Weather-Aware Traffic Prediction
This implements the complete research pipeline with extended training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../models')
from weather_aware_stgat import WeatherAwareSTGAT
from baseline_models import get_baseline_models, MLBaselines

class AdvancedTrainer:
    """Advanced trainer with comprehensive monitoring and analysis"""
    
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=0.01):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer with proper settings for deep learning
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=learning_rate * 0.01
        )
        
        # Loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Loss weights
        self.weather_loss_weight = 0.1
        self.l1_weight = 0.1
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Advanced metrics tracking
        self.epoch_metrics = {
            'train_mse': [],
            'train_mae': [],
            'val_mse': [],
            'val_mae': [],
            'val_r2': []
        }
        
    def compute_advanced_loss(self, predictions, targets, weather_logits=None, weather_targets=None):
        """Compute multi-component loss with weather awareness"""
        # Primary prediction losses
        mse_loss = self.mse_loss(predictions, targets)
        l1_loss = self.l1_loss(predictions, targets)
        
        # Combined prediction loss
        prediction_loss = mse_loss + self.l1_weight * l1_loss
        
        # Weather classification loss (auxiliary task)
        weather_loss = torch.tensor(0.0, device=self.device)
        if weather_logits is not None and weather_targets is not None:
            weather_logits_flat = weather_logits.view(-1, 4)
            weather_targets_flat = weather_targets.view(-1)
            weather_loss = self.cross_entropy_loss(weather_logits_flat, weather_targets_flat)
        
        # Total loss
        total_loss = prediction_loss + self.weather_loss_weight * weather_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'weather_loss': weather_loss.item(),
            'prediction_loss': prediction_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch):
        """Advanced training epoch with detailed monitoring"""
        self.model.train()
        total_losses = []
        mse_losses = []
        mae_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'weather' in str(type(self.model)).lower():
                predictions, weather_logits, _ = self.model(features)
                # Generate synthetic weather targets for auxiliary loss
                weather_targets = torch.randint(0, 4, (features.size(0), features.size(1)), device=self.device)
                loss, loss_components = self.compute_advanced_loss(predictions, targets, weather_logits, weather_targets)
            else:
                predictions = self.model(features)
                loss = self.mse_loss(predictions, targets)
                loss_components = {'total_loss': loss.item(), 'mse_loss': loss.item()}
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_losses.append(loss.item())
            mse_losses.append(loss_components['mse_loss'])
            
            # Calculate MAE
            with torch.no_grad():
                mae = torch.mean(torch.abs(predictions - targets)).item()
                mae_losses.append(mae)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{loss_components["mse_loss"]:.4f}',
                'mae': f'{mae:.4f}'
            })
        
        # Store epoch metrics
        avg_loss = np.mean(total_losses)
        avg_mse = np.mean(mse_losses)
        avg_mae = np.mean(mae_losses)
        
        self.epoch_metrics['train_mse'].append(avg_mse)
        self.epoch_metrics['train_mae'].append(avg_mae)
        
        return avg_loss, {'mse': avg_mse, 'mae': avg_mae}
    
    def validate_epoch(self, val_loader, epoch):
        """Advanced validation with comprehensive metrics"""
        self.model.eval()
        total_losses = []
        mse_losses = []
        mae_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation')
            for features, targets in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'weather' in str(type(self.model)).lower():
                    predictions, weather_logits, _ = self.model(features)
                    weather_targets = torch.randint(0, 4, (features.size(0), features.size(1)), device=self.device)
                    loss, loss_components = self.compute_advanced_loss(predictions, targets, weather_logits, weather_targets)
                else:
                    predictions = self.model(features)
                    loss = self.mse_loss(predictions, targets)
                    loss_components = {'total_loss': loss.item(), 'mse_loss': loss.item()}
                
                # Track metrics
                total_losses.append(loss.item())
                mse_losses.append(loss_components['mse_loss'])
                
                # Calculate MAE
                mae = torch.mean(torch.abs(predictions - targets)).item()
                mae_losses.append(mae)
                
                # Store for R² calculation
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{loss_components["mse_loss"]:.4f}',
                    'mae': f'{mae:.4f}'
                })
        
        # Calculate comprehensive metrics
        predictions_array = np.concatenate(all_predictions, axis=0)
        targets_array = np.concatenate(all_targets, axis=0)
        
        # R² calculation
        ss_res = np.sum((targets_array - predictions_array) ** 2)
        ss_tot = np.sum((targets_array - np.mean(targets_array)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Store epoch metrics
        avg_loss = np.mean(total_losses)
        avg_mse = np.mean(mse_losses)
        avg_mae = np.mean(mae_losses)
        
        self.epoch_metrics['val_mse'].append(avg_mse)
        self.epoch_metrics['val_mae'].append(avg_mae)
        self.epoch_metrics['val_r2'].append(r2)
        
        return avg_loss, {'mse': avg_mse, 'mae': avg_mae, 'r2': r2}
    
    def train(self, train_data, val_data, epochs=200, batch_size=64, patience=30, save_dir='../../results'):
        """Complete training with extended epochs and proper monitoring"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print(f"Full-scale training with {len(X_train):,} training samples")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        print(f"Device: {self.device}, Patience: {patience}")
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Track losses and learning rates
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(self.scheduler.get_last_lr()[0])
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"Train - Loss: {train_loss:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics
                }, os.path.join(save_dir, 'best_model_advanced.pth'))
                
                print(f"🎯 NEW BEST MODEL! Val Loss: {val_loss:.4f} (R²: {val_metrics['r2']:.4f})")
            else:
                self.patience_counter += 1
                print(f"⏳ Patience: {self.patience_counter}/{patience}")
                
            # Early stopping
            if self.patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'epoch_metrics': self.epoch_metrics
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save comprehensive training history
        training_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'epoch_metrics': self.epoch_metrics,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses),
            'final_lr': self.learning_rates[-1] if self.learning_rates else self.learning_rate
        }
        
        with open(os.path.join(save_dir, 'advanced_training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Create advanced training plots
        self.create_advanced_plots(save_dir)
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best R² Score: {max(self.epoch_metrics['val_r2']):.4f}")
        print(f"Total Epochs: {len(self.train_losses)}")
        
        return training_history
    
    def create_advanced_plots(self, save_dir):
        """Create comprehensive training visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        axes[0, 0].plot(epochs, self.val_losses, label='Validation Loss', color='red', alpha=0.7)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE comparison
        axes[0, 1].plot(epochs, self.epoch_metrics['train_mse'], label='Train MSE', color='blue', alpha=0.7)
        axes[0, 1].plot(epochs, self.epoch_metrics['val_mse'], label='Validation MSE', color='red', alpha=0.7)
        axes[0, 1].set_title('MSE Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAE comparison
        axes[0, 2].plot(epochs, self.epoch_metrics['train_mae'], label='Train MAE', color='blue', alpha=0.7)
        axes[0, 2].plot(epochs, self.epoch_metrics['val_mae'], label='Validation MAE', color='red', alpha=0.7)
        axes[0, 2].set_title('MAE Progress')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # R² Score progress
        axes[1, 0].plot(epochs, self.epoch_metrics['val_r2'], label='Validation R²', color='green', alpha=0.7)
        axes[1, 0].set_title('R² Score Progress')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        axes[1, 1].plot(epochs, self.learning_rates, color='orange', alpha=0.7)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Recent progress (last 50 epochs)
        recent_epochs = max(1, len(epochs) - 50)
        axes[1, 2].plot(epochs[recent_epochs:], self.train_losses[recent_epochs:], label='Train Loss', color='blue', alpha=0.7)
        axes[1, 2].plot(epochs[recent_epochs:], self.val_losses[recent_epochs:], label='Val Loss', color='red', alpha=0.7)
        axes[1, 2].set_title('Recent Training Progress (Last 50 Epochs)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Advanced Training Analysis - Weather-Aware STGAT', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'advanced_training_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

def run_full_training():
    """Run complete full-scale training and analysis"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting FULL-SCALE TRAINING on {device}")
    
    # Load complete dataset
    print("📂 Loading complete dataset...")
    X_train = np.load('../../data/processed/X_train.npy')
    y_train = np.load('../../data/processed/y_train.npy')
    X_val = np.load('../../data/processed/X_val.npy')
    y_val = np.load('../../data/processed/y_val.npy')
    X_test = np.load('../../data/processed/X_test.npy')
    y_test = np.load('../../data/processed/y_test.npy')
    
    print(f"📊 Dataset Statistics:")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_val.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Sequence length: {X_train.shape[1]}")
    print(f"  Feature dimensions: {X_train.shape[2]}")
    
    # Create Weather-Aware STGAT model with optimal configuration
    print("🏗️ Creating Weather-Aware STGAT model...")
    weather_model = WeatherAwareSTGAT(
        num_features=X_train.shape[-1],
        weather_features=5,
        hidden_dim=128,  # Increased for better capacity
        weather_dim=64,  # Increased weather representation
        num_layers=4,    # Deeper architecture
        num_heads=8,
        prediction_length=y_train.shape[-1],
        dropout=0.15,    # Slightly higher dropout for regularization
        num_nodes=1
    )
    
    total_params = sum(p.numel() for p in weather_model.parameters())
    trainable_params = sum(p.numel() for p in weather_model.parameters() if p.requires_grad)
    
    print(f"🧠 Model Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Create advanced trainer
    trainer = AdvancedTrainer(
        model=weather_model,
        device=device,
        learning_rate=0.0005,  # Slightly lower for stability
        weight_decay=0.01
    )
    
    # Full-scale training
    print("🎯 Starting full-scale training...")
    training_history = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=200,     # Extended training
        batch_size=128, # Larger batch size for stability
        patience=40,    # More patience for convergence
        save_dir='../../results/full_training'
    )
    
    return trainer, training_history

def main():
    """Main function for full training"""
    print("🌟 WEATHER-AWARE TRAFFIC PREDICTION - FULL ANALYSIS")
    print("="*80)
    
    trainer, history = run_full_training()
    
    print("✅ Full training completed!")
    return trainer, history

if __name__ == "__main__":
    trainer, history = main()