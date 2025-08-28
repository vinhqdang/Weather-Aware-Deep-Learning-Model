#!/usr/bin/env python3
"""
Dataset Analysis Script for Metro Interstate Traffic Volume
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset():
    """Analyze the Metro Interstate Traffic Volume dataset"""
    
    # Load the dataset
    print("Loading Metro Interstate Traffic Volume dataset...")
    df = pd.read_csv('../raw/Metro_Interstate_Traffic_Volume.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic information
    print("\n=== Dataset Info ===")
    print(df.info())
    
    print("\n=== First 5 rows ===")
    print(df.head())
    
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Check for missing values
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Traffic volume analysis
    print("\n=== Traffic Volume Analysis ===")
    print(f"Min traffic volume: {df['traffic_volume'].min()}")
    print(f"Max traffic volume: {df['traffic_volume'].max()}")
    print(f"Mean traffic volume: {df['traffic_volume'].mean():.2f}")
    print(f"Std traffic volume: {df['traffic_volume'].std():.2f}")
    
    # Weather features analysis
    weather_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
    print("\n=== Weather Features Analysis ===")
    for feature in weather_features:
        if feature in df.columns:
            print(f"{feature}: min={df[feature].min():.2f}, max={df[feature].max():.2f}, mean={df[feature].mean():.2f}")
    
    # Categorical features
    print("\n=== Categorical Features ===")
    categorical_features = ['weather_main', 'weather_description', 'holiday']
    for feature in categorical_features:
        if feature in df.columns:
            print(f"\n{feature} value counts:")
            print(df[feature].value_counts())
    
    # DateTime analysis
    if 'date_time' in df.columns:
        print("\n=== DateTime Analysis ===")
        df['date_time'] = pd.to_datetime(df['date_time'])
        print(f"Date range: {df['date_time'].min()} to {df['date_time'].max()}")
        
        # Extract temporal features
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month
        df['year'] = df['date_time'].dt.year
        
        print(f"Years covered: {sorted(df['year'].unique())}")
        print(f"Hours covered: {sorted(df['hour'].unique())}")
    
    # Weather conditions distribution
    print("\n=== Weather Conditions Distribution ===")
    if 'weather_main' in df.columns:
        weather_dist = df['weather_main'].value_counts()
        print(weather_dist)
        
        # Weather impact on traffic
        print("\n=== Weather Impact on Traffic ===")
        weather_traffic = df.groupby('weather_main')['traffic_volume'].agg(['mean', 'std', 'count'])
        print(weather_traffic)
    
    # Create visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Traffic volume distribution
    axes[0, 0].hist(df['traffic_volume'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Traffic Volume Distribution')
    axes[0, 0].set_xlabel('Traffic Volume')
    axes[0, 0].set_ylabel('Frequency')
    
    # Traffic volume by hour
    if 'hour' in df.columns:
        hourly_traffic = df.groupby('hour')['traffic_volume'].mean()
        axes[0, 1].plot(hourly_traffic.index, hourly_traffic.values, marker='o')
        axes[0, 1].set_title('Average Traffic Volume by Hour')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Average Traffic Volume')
        axes[0, 1].grid(True)
    
    # Weather main distribution
    if 'weather_main' in df.columns:
        weather_counts = df['weather_main'].value_counts()
        axes[1, 0].pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Weather Conditions Distribution')
    
    # Temperature vs Traffic Volume scatter
    if 'temp' in df.columns:
        # Sample for better visualization
        sample_df = df.sample(n=min(5000, len(df)))
        axes[1, 1].scatter(sample_df['temp'], sample_df['traffic_volume'], alpha=0.5, s=1)
        axes[1, 1].set_title('Temperature vs Traffic Volume')
        axes[1, 1].set_xlabel('Temperature (K)')
        axes[1, 1].set_ylabel('Traffic Volume')
    
    plt.tight_layout()
    plt.savefig('../processed/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save processed insights
    insights = {
        'dataset_shape': df.shape,
        'date_range': f"{df['date_time'].min()} to {df['date_time'].max()}" if 'date_time' in df.columns else "N/A",
        'traffic_volume_stats': {
            'min': float(df['traffic_volume'].min()),
            'max': float(df['traffic_volume'].max()),
            'mean': float(df['traffic_volume'].mean()),
            'std': float(df['traffic_volume'].std())
        },
        'weather_conditions': df['weather_main'].value_counts().to_dict() if 'weather_main' in df.columns else {},
        'missing_values': missing_values[missing_values > 0].to_dict(),
        'columns': list(df.columns)
    }
    
    # Save insights as JSON
    import json
    with open('../processed/dataset_insights.json', 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    
    print(f"\n=== Analysis Complete ===")
    print("Results saved to:")
    print("- ../processed/dataset_analysis.png")
    print("- ../processed/dataset_insights.json")
    
    return df, insights

if __name__ == "__main__":
    df, insights = analyze_dataset()