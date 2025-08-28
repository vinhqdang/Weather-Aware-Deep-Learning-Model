"""
Create Results Analysis and Visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def create_demo_results():
    """Create demo results from the output"""
    
    # Results from the demo run
    results = {
        'random_forest': {
            'mse': 0.0182,
            'rmse': 0.1350,
            'mae': 0.0952,
            'mape': 50.27,
            'r2': 0.7782
        },
        'weather_aware_stgat': {
            'mse': 0.0225,  # Calculated from RMSE
            'rmse': 0.1502,
            'mae': 0.1135,
            'mape': 68.49,
            'r2': 0.7253
        },
        'lstm': {
            'mse': 0.0267,  # Calculated from RMSE
            'rmse': 0.1634,
            'mae': 0.1213,
            'mape': 62.99,
            'r2': 0.6750
        },
        'linear_regression': {
            'mse': 0.1019,
            'rmse': 0.3192,
            'mae': 0.1847,
            'mape': 107.56,
            'r2': -0.2401
        }
    }
    
    # Calculate improvement
    baseline_models = {k: v for k, v in results.items() if k != 'weather_aware_stgat'}
    best_baseline_rmse = min([v['rmse'] for v in baseline_models.values()])
    weather_rmse = results['weather_aware_stgat']['rmse']
    improvement = ((best_baseline_rmse - weather_rmse) / best_baseline_rmse) * 100
    
    demo_results = {
        'model_comparison': results,
        'improvement_percentage': float(improvement),
        'best_baseline_model': 'random_forest',
        'note': 'Quick demo with reduced data (5000 train, 1000 test) and 5 epochs',
        'experiment_info': {
            'dataset': 'Metro Interstate Traffic Volume',
            'train_samples': 5000,
            'test_samples': 1000,
            'sequence_length': 12,
            'num_features': 26,
            'prediction_length': 12,
            'training_epochs': 5
        }
    }
    
    return demo_results

def create_visualizations(results):
    """Create comprehensive visualizations"""
    
    model_comparison = results['model_comparison']
    models = list(model_comparison.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for different models
    colors = ['lightcoral' if 'linear' in m else 'skyblue' if 'lstm' in m else 'lightgreen' if 'random' in m else 'orange' for m in models]
    
    # RMSE comparison
    rmse_values = [model_comparison[m]['rmse'] for m in models]
    bars1 = axes[0, 0].bar(models, rmse_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars1, rmse_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE comparison
    mae_values = [model_comparison[m]['mae'] for m in models]
    bars2 = axes[0, 1].bar(models, mae_values, color=colors, alpha=0.7)
    axes[0, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, mae_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # R² comparison
    r2_values = [model_comparison[m]['r2'] for m in models]
    bars3 = axes[1, 0].bar(models, r2_values, color=colors, alpha=0.7)
    axes[1, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for bar, value in zip(bars3, r2_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if value > 0 else -0.05), 
                       f'{value:.4f}', ha='center', va='bottom' if value > 0 else 'top', fontweight='bold')
    
    # MAPE comparison
    mape_values = [model_comparison[m]['mape'] for m in models]
    bars4 = axes[1, 1].bar(models, mape_values, color=colors, alpha=0.7)
    axes[1, 1].set_title('MAPE Comparison (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars4, mape_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Weather-Aware Traffic Prediction Model Comparison\n(Quick Demo Results)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../../results/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create improvement visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Sort models by RMSE performance
    sorted_models = sorted(model_comparison.items(), key=lambda x: x[1]['rmse'])
    sorted_names = [item[0] for item in sorted_models]
    sorted_rmse = [item[1]['rmse'] for item in sorted_models]
    
    # Color code: best model in green, weather-aware in blue, others in gray
    bar_colors = []
    for name in sorted_names:
        if name == 'weather_aware_stgat':
            bar_colors.append('steelblue')
        elif name == sorted_names[0]:  # Best performing
            bar_colors.append('forestgreen')
        else:
            bar_colors.append('lightgray')
    
    bars = ax.bar(range(len(sorted_names)), sorted_rmse, color=bar_colors, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Models (sorted by RMSE performance)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Traffic Volume Prediction Performance\nWeather-Aware Deep Learning vs Baselines', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels([name.replace('_', ' ').title() for name in sorted_names], rotation=45)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_rmse)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
               f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='forestgreen', alpha=0.8, label='Best Baseline'),
        plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.8, label='Weather-Aware STGAT'),
        plt.Rectangle((0,0),1,1, color='lightgray', alpha=0.8, label='Other Baselines')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('../../results/performance_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations created:")
    print("- ../../results/comprehensive_model_comparison.png")
    print("- ../../results/performance_ranking.png")

def create_results_summary():
    """Create a comprehensive results summary"""
    
    results = create_demo_results()
    
    # Create results directory
    os.makedirs('../../results', exist_ok=True)
    
    # Save results as JSON
    with open('../../results/demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    create_visualizations(results)
    
    # Create summary report
    report = f"""
# Weather-Aware Traffic Prediction Results Summary

## Experiment Overview
- **Dataset**: Metro Interstate Traffic Volume (Kaggle)
- **Training Samples**: {results['experiment_info']['train_samples']:,}
- **Test Samples**: {results['experiment_info']['test_samples']:,}
- **Sequence Length**: {results['experiment_info']['sequence_length']} hours
- **Features**: {results['experiment_info']['num_features']} (temporal + weather)
- **Prediction Horizon**: {results['experiment_info']['prediction_length']} hours

## Model Performance Comparison

| Model | RMSE | MAE | MAPE (%) | R² Score |
|-------|------|-----|----------|----------|
"""
    
    # Sort models by RMSE
    sorted_models = sorted(results['model_comparison'].items(), key=lambda x: x[1]['rmse'])
    
    for model_name, metrics in sorted_models:
        model_display = model_name.replace('_', ' ').title()
        report += f"| {model_display} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['mape']:.2f} | {metrics['r2']:.4f} |\n"
    
    # Add key findings
    best_baseline = results['best_baseline_model'].replace('_', ' ').title()
    weather_rmse = results['model_comparison']['weather_aware_stgat']['rmse']
    best_baseline_rmse = results['model_comparison'][results['best_baseline_model']]['rmse']
    
    if weather_rmse < best_baseline_rmse:
        improvement_text = f"Weather-Aware STGAT achieved {abs(results['improvement_percentage']):.2f}% better RMSE than the best baseline"
    else:
        improvement_text = f"Weather-Aware STGAT performed {abs(results['improvement_percentage']):.2f}% worse than the best baseline in this quick demo"
    
    report += f"""
## Key Findings

### Performance Analysis
- **Best Overall Model**: {sorted_models[0][0].replace('_', ' ').title()} (RMSE: {sorted_models[0][1]['rmse']:.4f})
- **Weather-Aware STGAT Performance**: {improvement_text}
- **Baseline Comparison**: {best_baseline} was the strongest baseline model

### Weather-Aware Features Impact
The Weather-Aware STGAT model incorporates:
- **Dynamic Weather Attention**: Adapts predictions based on weather conditions
- **Multi-Scale Temporal Processing**: Captures different time scale patterns
- **Weather Feature Encoding**: Transforms raw weather data into meaningful representations

### Model Complexity
- **Weather-Aware STGAT**: 187,992 parameters
- **LSTM Baseline**: ~177,000 parameters
- **Random Forest**: Tree-based ensemble
- **Linear Regression**: Simplest baseline

## Technical Implementation

### Weather Features Used
- Temperature (Celsius, scaled)
- Precipitation (rain_1h, snow_1h, scaled)
- Cloud coverage (scaled)
- Weather severity index (computed)
- Weather type classifications

### Data Preprocessing
- Sequence length: 12 hours input → 12 hours prediction
- Feature engineering: 26 features including temporal and weather
- Normalization: MinMax scaling for traffic, StandardScaler for weather
- Temporal split: 70% train, 20% validation, 10% test

## Notes
⚠️ **Important**: This is a quick demonstration with reduced data size and training epochs. 
Full-scale training with the complete dataset and extended training would likely show different results.

The Random Forest performed exceptionally well in this demo, possibly due to:
1. Smaller dataset size favoring tree-based methods
2. Limited training epochs for deep learning models
3. Feature engineering well-suited for tree-based algorithms

## Future Improvements
- Train with full dataset and extended epochs
- Hyperparameter tuning
- Cross-validation for robust evaluation
- Analysis of weather-specific performance
- Attention visualization and interpretability analysis
"""
    
    # Save report
    with open('../../results/results_summary.md', 'w') as f:
        f.write(report)
    
    print("Results summary created:")
    print("- ../../results/demo_results.json")
    print("- ../../results/results_summary.md")
    
    return results

def main():
    """Main function to generate all results"""
    print("Creating comprehensive results analysis...")
    
    results = create_results_summary()
    
    print(f"\n" + "="*60)
    print("WEATHER-AWARE TRAFFIC PREDICTION - DEMO RESULTS")
    print("="*60)
    
    # Print key results
    sorted_models = sorted(results['model_comparison'].items(), key=lambda x: x[1]['rmse'])
    
    print(f"{'Model':<25} {'RMSE':<8} {'MAE':<8} {'R² Score':<10}")
    print("-" * 55)
    
    for model_name, metrics in sorted_models:
        model_display = model_name.replace('_', ' ').title()
        print(f"{model_display:<25} {metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} {metrics['r2']:<10.4f}")
    
    print(f"\nBest Model: {sorted_models[0][0].replace('_', ' ').title()}")
    print(f"Weather-Aware STGAT Performance: {results['improvement_percentage']:.2f}% vs best baseline")
    
    print(f"\n⚠️  Note: Quick demo results with limited data and training.")
    print("Complete results would require full dataset and extended training.")
    
    return results

if __name__ == "__main__":
    results = main()