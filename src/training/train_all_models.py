"""
Complete Training and Evaluation Script for All Models
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

class SimpleTrainer:
    """Simple trainer for baseline models"""
    
    def __init__(self, model, device='cuda', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_data, val_data, epochs=50, batch_size=64, patience=10):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
            
            self.scheduler.step()
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        return {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val_loss}
    
    def evaluate(self, test_data, batch_size=64):
        X_test, y_test = test_data
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                predictions = self.model(features)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mse': mse,
            'rmse': rmse, 
            'mae': mae,
            'mape': mape,
            'r2': r2
        }, predictions, targets

def train_and_evaluate_all_models():
    """Train and evaluate all models"""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading preprocessed data...")
    X_train = np.load('../../data/processed/X_train.npy')
    y_train = np.load('../../data/processed/y_train.npy')
    X_val = np.load('../../data/processed/X_val.npy')
    y_val = np.load('../../data/processed/y_val.npy')
    X_test = np.load('../../data/processed/X_test.npy')
    y_test = np.load('../../data/processed/y_test.npy')
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Results storage
    all_results = {}
    all_predictions = {}
    
    # Create results directory
    os.makedirs('../../results', exist_ok=True)
    
    # 1. Train ML Baselines
    print("\n" + "="*50)
    print("Training ML Baseline Models")
    print("="*50)
    
    ml_baselines = MLBaselines()
    ml_train_results = ml_baselines.train(X_train, y_train, X_val, y_val)
    ml_test_results, ml_predictions = ml_baselines.evaluate(X_test, y_test)
    
    all_results.update(ml_test_results)
    all_predictions.update(ml_predictions)
    
    # 2. Train Deep Learning Baselines
    print("\n" + "="*50)
    print("Training Deep Learning Baseline Models")
    print("="*50)
    
    baseline_models = get_baseline_models(X_train.shape[-1], y_train.shape[-1])
    
    for model_name, model in baseline_models.items():
        print(f"\nTraining {model_name.upper()}...")
        
        trainer = SimpleTrainer(model, device=device, learning_rate=0.001)
        
        # Train
        training_history = trainer.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            patience=10
        )
        
        # Evaluate
        test_metrics, predictions, targets = trainer.evaluate(
            test_data=(X_test, y_test),
            batch_size=64
        )
        
        print(f"{model_name.upper()} Results:")
        print(f"  MSE: {test_metrics['mse']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  R²: {test_metrics['r2']:.4f}")
        
        all_results[model_name] = test_metrics
        all_predictions[model_name] = predictions
        
        # Save model
        torch.save(model.state_dict(), f'../../results/{model_name}_model.pth')
    
    # 3. Train Weather-Aware STGAT
    print("\n" + "="*50)
    print("Training Weather-Aware STGAT Model")
    print("="*50)
    
    # Import the trainer (need tqdm for progress bars)
    from trainer import WeatherAwareTrainer
    
    # Create Weather-Aware STGAT model
    weather_model = WeatherAwareSTGAT(
        num_features=X_train.shape[-1],
        weather_features=5,
        hidden_dim=128,
        weather_dim=32,
        num_layers=3,
        num_heads=8,
        prediction_length=y_train.shape[-1],
        dropout=0.1,
        num_nodes=1
    )
    
    print(f"Weather-Aware STGAT parameters: {sum(p.numel() for p in weather_model.parameters()):,}")
    
    # Create trainer
    weather_trainer = WeatherAwareTrainer(
        model=weather_model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    # Train
    training_history = weather_trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        patience=15,
        save_dir='../../results'
    )
    
    # Evaluate
    weather_metrics, weather_predictions, weather_targets = weather_trainer.evaluate(
        test_data=(X_test, y_test),
        batch_size=64
    )
    
    all_results['weather_aware_stgat'] = weather_metrics
    all_predictions['weather_aware_stgat'] = weather_predictions
    
    # 4. Analyze and Compare Results
    print("\n" + "="*50)
    print("FINAL RESULTS COMPARISON")
    print("="*50)
    
    # Sort models by RMSE performance
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['rmse'])
    
    print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'MAPE':<8} {'R²':<8}")
    print("-" * 60)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<20} {metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} {metrics['mape']:<8.2f} {metrics['r2']:<8.4f}")
    
    # Calculate improvements
    best_baseline_rmse = min([v['rmse'] for k, v in all_results.items() if k != 'weather_aware_stgat'])
    weather_rmse = all_results['weather_aware_stgat']['rmse']
    improvement = ((best_baseline_rmse - weather_rmse) / best_baseline_rmse) * 100
    
    print(f"\nWeather-Aware STGAT Improvement: {improvement:.2f}% better RMSE than best baseline")
    
    # 5. Save All Results
    final_results = {
        'model_comparison': all_results,
        'improvement_percentage': improvement,
        'best_baseline_model': min(all_results.items(), key=lambda x: x[1]['rmse'] if x[0] != 'weather_aware_stgat' else float('inf'))[0],
        'experiment_info': {
            'dataset': 'Metro Interstate Traffic Volume',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'sequence_length': X_train.shape[1],
            'num_features': X_train.shape[2],
            'prediction_length': y_train.shape[1],
            'device': device
        }
    }
    
    # Save results
    with open('../../results/comprehensive_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save predictions
    np.save('../../results/all_predictions.npy', all_predictions)
    np.save('../../results/test_targets.npy', y_test)
    
    print(f"\nAll results saved to ../../results/")
    print("Training and evaluation completed!")
    
    return final_results, all_predictions

def create_results_visualization():
    """Create comprehensive results visualization"""
    
    # Load results
    with open('../../results/comprehensive_results.json', 'r') as f:
        results = json.load(f)
    
    model_comparison = results['model_comparison']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(model_comparison.keys())
    rmse_values = [model_comparison[m]['rmse'] for m in models]
    mae_values = [model_comparison[m]['mae'] for m in models]
    mape_values = [model_comparison[m]['mape'] for m in models]
    r2_values = [model_comparison[m]['r2'] for m in models]
    
    # RMSE comparison
    axes[0, 0].bar(models, rmse_values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('RMSE Comparison')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    axes[0, 1].bar(models, mae_values, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('MAE Comparison')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # MAPE comparison
    axes[1, 0].bar(models, mape_values, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('MAPE Comparison (%)')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[1, 1].bar(models, r2_values, color='orange', alpha=0.7)
    axes[1, 1].set_title('R² Comparison')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../../results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Results visualization saved to ../../results/model_comparison.png")

def main():
    """Main function"""
    print("Starting comprehensive model training and evaluation...")
    
    # Train and evaluate all models
    final_results, all_predictions = train_and_evaluate_all_models()
    
    # Create visualizations
    create_results_visualization()
    
    print("\nExperiment completed successfully!")
    print(f"Weather-Aware STGAT achieved {final_results['improvement_percentage']:.2f}% improvement over best baseline")
    
    return final_results

if __name__ == "__main__":
    results = main()