"""
Quick Demo Training Script for Weather-Aware Traffic Prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../models')
from weather_aware_stgat import WeatherAwareSTGAT
from baseline_models import get_baseline_models, MLBaselines

def quick_train_model(model, train_data, val_data, epochs=5, device='cuda'):
    """Quick training function"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            if hasattr(model, 'forward') and 'weather' in str(type(model)).lower():
                predictions, _, _ = model(features)
            else:
                predictions = model(features)
            
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                if hasattr(model, 'forward') and 'weather' in str(type(model)).lower():
                    predictions, _, _ = model(features)
                else:
                    predictions = model(features)
                
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_data, device='cuda'):
    """Evaluate model on test data"""
    X_test, y_test = test_data
    
    model = model.to(device)
    model.eval()
    
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            
            if hasattr(model, 'forward') and 'weather' in str(type(model)).lower():
                predictions, _, _ = model(features)
            else:
                predictions = model(features)
            
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

def main():
    """Quick demo training"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data (smaller subset for demo)
    print("Loading data...")
    X_train = np.load('../../data/processed/X_train.npy')[:5000]  # Use subset
    y_train = np.load('../../data/processed/y_train.npy')[:5000]
    X_val = np.load('../../data/processed/X_val.npy')[:1000]
    y_val = np.load('../../data/processed/y_val.npy')[:1000]
    X_test = np.load('../../data/processed/X_test.npy')[:1000]
    y_test = np.load('../../data/processed/y_test.npy')[:1000]
    
    print(f"Demo data: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    results = {}
    
    # 1. Train ML Baselines (quick)
    print("\n" + "="*50)
    print("Training ML Baselines (Quick Demo)")
    print("="*50)
    
    ml_baselines = MLBaselines()
    ml_train_results = ml_baselines.train(X_train, y_train, X_val, y_val)
    ml_test_results, ml_predictions = ml_baselines.evaluate(X_test, y_test)
    
    results.update(ml_test_results)
    
    # 2. Train one deep learning baseline (LSTM)
    print("\n" + "="*50)
    print("Training LSTM Baseline (Quick Demo)")
    print("="*50)
    
    from baseline_models import LSTMBaseline
    
    lstm_model = LSTMBaseline(
        input_dim=X_train.shape[-1],
        hidden_dim=64,  # Smaller for demo
        num_layers=1,
        prediction_length=y_train.shape[-1],
        dropout=0.1
    )
    
    train_losses, val_losses = quick_train_model(
        lstm_model, 
        (X_train, y_train), 
        (X_val, y_val), 
        epochs=5,
        device=device
    )
    
    lstm_metrics, lstm_predictions, _ = evaluate_model(lstm_model, (X_test, y_test), device)
    results['lstm'] = lstm_metrics
    
    print(f"LSTM Results: RMSE={lstm_metrics['rmse']:.4f}, MAE={lstm_metrics['mae']:.4f}, R²={lstm_metrics['r2']:.4f}")
    
    # 3. Train Weather-Aware STGAT (quick)
    print("\n" + "="*50)
    print("Training Weather-Aware STGAT (Quick Demo)")
    print("="*50)
    
    weather_model = WeatherAwareSTGAT(
        num_features=X_train.shape[-1],
        weather_features=5,
        hidden_dim=64,  # Smaller for demo
        weather_dim=16,
        num_layers=2,
        num_heads=4,
        prediction_length=y_train.shape[-1],
        dropout=0.1,
        num_nodes=1
    )
    
    print(f"Weather-Aware STGAT parameters: {sum(p.numel() for p in weather_model.parameters()):,}")
    
    train_losses, val_losses = quick_train_model(
        weather_model,
        (X_train, y_train),
        (X_val, y_val),
        epochs=5,
        device=device
    )
    
    weather_metrics, weather_predictions, _ = evaluate_model(
        weather_model, 
        (X_test, y_test), 
        device
    )
    results['weather_aware_stgat'] = weather_metrics
    
    print(f"Weather-Aware STGAT Results: RMSE={weather_metrics['rmse']:.4f}, MAE={weather_metrics['mae']:.4f}, R²={weather_metrics['r2']:.4f}")
    
    # 4. Compare Results
    print("\n" + "="*50)
    print("QUICK DEMO RESULTS COMPARISON")
    print("="*50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
    
    print(f"{'Model':<25} {'RMSE':<8} {'MAE':<8} {'MAPE':<8} {'R²':<8}")
    print("-" * 65)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} {metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} {metrics['mape']:<8.2f} {metrics['r2']:<8.4f}")
    
    # Calculate improvement
    best_baseline_rmse = min([v['rmse'] for k, v in results.items() if k != 'weather_aware_stgat'])
    weather_rmse = results['weather_aware_stgat']['rmse']
    improvement = ((best_baseline_rmse - weather_rmse) / best_baseline_rmse) * 100
    
    print(f"\nWeather-Aware STGAT Improvement: {improvement:.2f}% better RMSE than best baseline")
    
    # Save results
    os.makedirs('../../results', exist_ok=True)
    demo_results = {
        'model_comparison': results,
        'improvement_percentage': improvement,
        'note': 'Quick demo with reduced data and epochs'
    }
    
    with open('../../results/demo_results.json', 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    # Create simple visualization
    models = list(results.keys())
    rmse_values = [results[m]['rmse'] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, rmse_values, color=['lightcoral', 'skyblue', 'lightgreen'][:len(models)])
    plt.title('Model Comparison - RMSE (Quick Demo)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../../results/demo_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDemo completed! Results saved to ../../results/")
    print("Note: This is a quick demo with reduced data and training epochs.")
    print("For full results, run the complete training pipeline.")
    
    return demo_results

if __name__ == "__main__":
    results = main()