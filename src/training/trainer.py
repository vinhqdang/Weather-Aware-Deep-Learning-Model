"""
Training Pipeline for Weather-Aware Traffic Prediction
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

class WeatherAwareTrainer:
    """
    Trainer class for Weather-Aware STGAT model
    """
    
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=0.01):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=learning_rate * 0.01
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
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def compute_loss(self, predictions, targets, weather_logits=None, weather_targets=None):
        """
        Compute multi-component loss
        
        Args:
            predictions: Model predictions [batch_size, prediction_length]
            targets: True targets [batch_size, prediction_length]
            weather_logits: Weather classification logits [batch_size, seq_len, 4]
            weather_targets: Weather classification targets [batch_size, seq_len]
        
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        # Primary prediction losses
        mse_loss = self.mse_loss(predictions, targets)
        l1_loss = self.l1_loss(predictions, targets)
        
        # Combined prediction loss
        prediction_loss = mse_loss + self.l1_weight * l1_loss
        
        # Weather classification loss (auxiliary task)
        weather_loss = torch.tensor(0.0, device=self.device)
        if weather_logits is not None and weather_targets is not None:
            # Reshape for cross entropy
            weather_logits_flat = weather_logits.view(-1, 4)
            weather_targets_flat = weather_targets.view(-1)
            weather_loss = self.cross_entropy_loss(weather_logits_flat, weather_targets_flat)
        
        # Total loss
        total_loss = prediction_loss + self.weather_loss_weight * weather_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'weather_loss': weather_loss.item(),
            'prediction_loss': prediction_loss.item()
        }
        
        return total_loss, loss_components
    
    def train_epoch(self, train_loader, weather_targets=None):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            weather_targets: Optional weather classification targets
        
        Returns:
            avg_loss: Average training loss for the epoch
            loss_components: Average of individual loss components
        """
        self.model.train()
        total_losses = []
        all_loss_components = {
            'total_loss': [],
            'mse_loss': [],
            'l1_loss': [],
            'weather_loss': [],
            'prediction_loss': []
        }
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions, weather_logits, _ = self.model(features)
            
            # Generate weather targets (simplified - in practice should be from data)
            batch_weather_targets = None
            if weather_targets is not None:
                batch_weather_targets = weather_targets[batch_idx * features.size(0):(batch_idx + 1) * features.size(0)]
                batch_weather_targets = batch_weather_targets.to(self.device)
            
            # Compute loss
            loss, loss_components = self.compute_loss(
                predictions, targets, weather_logits, batch_weather_targets
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            total_losses.append(loss.item())
            for key, value in loss_components.items():
                all_loss_components[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average losses
        avg_loss = np.mean(total_losses)
        avg_loss_components = {key: np.mean(values) for key, values in all_loss_components.items()}
        
        return avg_loss, avg_loss_components
    
    def validate_epoch(self, val_loader, weather_targets=None):
        """
        Validate for one epoch
        
        Args:
            val_loader: DataLoader for validation data
            weather_targets: Optional weather classification targets
        
        Returns:
            avg_loss: Average validation loss
            loss_components: Average of individual loss components
        """
        self.model.eval()
        total_losses = []
        all_loss_components = {
            'total_loss': [],
            'mse_loss': [],
            'l1_loss': [],
            'weather_loss': [],
            'prediction_loss': []
        }
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (features, targets) in enumerate(pbar):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions, weather_logits, _ = self.model(features)
                
                # Generate weather targets
                batch_weather_targets = None
                if weather_targets is not None:
                    batch_weather_targets = weather_targets[batch_idx * features.size(0):(batch_idx + 1) * features.size(0)]
                    batch_weather_targets = batch_weather_targets.to(self.device)
                
                # Compute loss
                loss, loss_components = self.compute_loss(
                    predictions, targets, weather_logits, batch_weather_targets
                )
                
                # Track losses
                total_losses.append(loss.item())
                for key, value in loss_components.items():
                    all_loss_components[key].append(value)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average losses
        avg_loss = np.mean(total_losses)
        avg_loss_components = {key: np.mean(values) for key, values in all_loss_components.items()}
        
        return avg_loss, avg_loss_components
    
    def train(self, train_data, val_data, epochs=100, batch_size=64, patience=20, save_dir='../../results'):
        """
        Complete training loop
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            save_dir: Directory to save results
        
        Returns:
            training_history: Dictionary containing training history
        """
        # Create data loaders
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training Weather-Aware STGAT for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}, Device: {self.device}")
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_components = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_components = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Track losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train MSE: {train_components['mse_loss']:.4f}, Val MSE: {val_components['mse_loss']:.4f}")
            print(f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
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
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save training history
        training_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses)
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        print(f"\nTraining completed! Best val loss: {self.best_val_loss:.4f}")
        
        return training_history
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses[-50:], label='Train Loss (Last 50)')
        plt.plot(self.val_losses[-50:], label='Val Loss (Last 50)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, test_data, batch_size=64):
        """
        Evaluate model on test data
        
        Args:
            test_data: Tuple of (X_test, y_test)
            batch_size: Batch size for evaluation
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        X_test, y_test = test_data
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(test_loader, desc='Testing'):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions, _, _ = self.model(features)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        print(f"Test Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")
        
        return metrics, predictions, targets

def main():
    """Main training function"""
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('../../data/processed/X_train.npy')
    y_train = np.load('../../data/processed/y_train.npy')
    X_val = np.load('../../data/processed/X_val.npy')
    y_val = np.load('../../data/processed/y_val.npy')
    X_test = np.load('../../data/processed/X_test.npy')
    y_test = np.load('../../data/processed/y_test.npy')
    
    print(f"Data loaded:")
    print(f"Train: {X_train.shape} -> {y_train.shape}")
    print(f"Val: {X_val.shape} -> {y_val.shape}")
    print(f"Test: {X_test.shape} -> {y_test.shape}")
    
    # Create model
    model = WeatherAwareSTGAT(
        num_features=X_train.shape[-1],  # 26 features
        weather_features=5,  # 5 weather features
        hidden_dim=128,
        weather_dim=32,
        num_layers=3,
        num_heads=8,
        prediction_length=y_train.shape[-1],  # 12 predictions
        dropout=0.1,
        num_nodes=1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = WeatherAwareTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    # Train model
    training_history = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        patience=20,
        save_dir='../../results'
    )
    
    # Evaluate on test set
    test_metrics, test_predictions, test_targets = trainer.evaluate(
        test_data=(X_test, y_test),
        batch_size=64
    )
    
    # Save test results
    results = {
        'test_metrics': test_metrics,
        'training_history': training_history,
        'model_info': {
            'num_features': X_train.shape[-1],
            'hidden_dim': 128,
            'num_layers': 3,
            'num_heads': 8,
            'prediction_length': y_train.shape[-1]
        }
    }
    
    with open('../../results/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.save('../../results/test_predictions.npy', test_predictions)
    np.save('../../results/test_targets.npy', test_targets)
    
    print("Training and evaluation completed!")
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()