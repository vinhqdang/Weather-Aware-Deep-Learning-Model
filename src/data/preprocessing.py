"""
Data Preprocessing Pipeline for Weather-Aware Traffic Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TrafficDataPreprocessor:
    """Preprocessing pipeline for traffic and weather data"""
    
    def __init__(self, sequence_length=12, prediction_length=12, test_size=0.2, val_size=0.2):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.test_size = test_size
        self.val_size = val_size
        
        self.traffic_scaler = MinMaxScaler()
        self.weather_scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load and basic clean the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Convert datetime
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        # Sort by datetime
        df = df.sort_values('date_time').reset_index(drop=True)
        
        print(f"Dataset loaded: {df.shape}")
        return df
    
    def create_temporal_features(self, df):
        """Create temporal features from datetime"""
        print("Creating temporal features...")
        
        df = df.copy()
        
        # Basic temporal features
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['month'] = df['date_time'].dt.month
        df['year'] = df['date_time'].dt.year
        df['day_of_year'] = df['date_time'].dt.dayofyear
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Holiday indicator (binary)
        df['is_holiday'] = (~df['holiday'].isna()).astype(int)
        
        # Rush hour indicators
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_weather_features(self, df):
        """Enhanced weather feature engineering"""
        print("Creating weather features...")
        
        df = df.copy()
        
        # Temperature features
        df['temp_celsius'] = df['temp'] - 273.15  # Convert from Kelvin
        df['temp_feels_like'] = df['temp_celsius']  # Simplified
        
        # Temperature categories
        df['temp_very_cold'] = (df['temp_celsius'] < -10).astype(int)
        df['temp_cold'] = ((df['temp_celsius'] >= -10) & (df['temp_celsius'] < 0)).astype(int)
        df['temp_cool'] = ((df['temp_celsius'] >= 0) & (df['temp_celsius'] < 15)).astype(int)
        df['temp_mild'] = ((df['temp_celsius'] >= 15) & (df['temp_celsius'] < 25)).astype(int)
        df['temp_warm'] = ((df['temp_celsius'] >= 25) & (df['temp_celsius'] < 35)).astype(int)
        df['temp_hot'] = (df['temp_celsius'] >= 35).astype(int)
        
        # Precipitation features
        df['has_rain'] = (df['rain_1h'] > 0).astype(int)
        df['has_snow'] = (df['snow_1h'] > 0).astype(int)
        df['has_precipitation'] = ((df['rain_1h'] > 0) | (df['snow_1h'] > 0)).astype(int)
        
        # Precipitation intensity
        df['rain_light'] = ((df['rain_1h'] > 0) & (df['rain_1h'] <= 2.5)).astype(int)
        df['rain_moderate'] = ((df['rain_1h'] > 2.5) & (df['rain_1h'] <= 10)).astype(int)
        df['rain_heavy'] = (df['rain_1h'] > 10).astype(int)
        
        # Snow intensity
        df['snow_light'] = ((df['snow_1h'] > 0) & (df['snow_1h'] <= 0.1)).astype(int)
        df['snow_moderate'] = ((df['snow_1h'] > 0.1) & (df['snow_1h'] <= 0.3)).astype(int)
        df['snow_heavy'] = (df['snow_1h'] > 0.3).astype(int)
        
        # Cloud cover categories
        df['sky_clear'] = (df['clouds_all'] <= 10).astype(int)
        df['sky_partly_cloudy'] = ((df['clouds_all'] > 10) & (df['clouds_all'] <= 50)).astype(int)
        df['sky_mostly_cloudy'] = ((df['clouds_all'] > 50) & (df['clouds_all'] <= 90)).astype(int)
        df['sky_overcast'] = (df['clouds_all'] > 90).astype(int)
        
        # Weather severity index
        weather_severity = 0
        weather_severity += df['rain_1h'] / 10  # Rain impact
        weather_severity += df['snow_1h'] * 20  # Snow has higher impact
        weather_severity += (df['clouds_all'] / 100) * 0.5  # Cloud impact
        weather_severity += np.abs(df['temp_celsius'] - 20) / 30  # Temperature deviation impact
        
        df['weather_severity'] = weather_severity
        
        # Adverse weather indicator
        df['adverse_weather'] = (
            (df['rain_1h'] > 1) | 
            (df['snow_1h'] > 0) | 
            (df['clouds_all'] > 80) |
            (df['temp_celsius'] < -5) |
            (df['temp_celsius'] > 35)
        ).astype(int)
        
        # Encode categorical weather features
        weather_main_encoder = LabelEncoder()
        df['weather_main_encoded'] = weather_main_encoder.fit_transform(df['weather_main'])
        self.label_encoders['weather_main'] = weather_main_encoder
        
        # One-hot encode major weather types
        weather_types = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist', 'Fog', 'Drizzle', 'Thunderstorm']
        for weather_type in weather_types:
            df[f'weather_{weather_type.lower()}'] = (df['weather_main'] == weather_type).astype(int)
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        df = df.copy()
        
        # Handle missing holiday values (most are non-holidays)
        df['holiday'] = df['holiday'].fillna('None')
        
        # Forward fill for weather data (reasonable for hourly data)
        weather_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
        for col in weather_cols:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Handle any remaining missing values
        df = df.fillna(0)
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        print("Scaling features...")
        
        df = df.copy()
        
        # Scale traffic volume (target variable)
        if fit:
            df['traffic_volume_scaled'] = self.traffic_scaler.fit_transform(df[['traffic_volume']])
        else:
            df['traffic_volume_scaled'] = self.traffic_scaler.transform(df[['traffic_volume']])
        
        # Define feature groups for scaling
        weather_features = [
            'temp_celsius', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_severity'
        ]
        
        temporal_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        # Scale weather features
        for feature in weather_features:
            if feature in df.columns:
                if fit:
                    scaler = StandardScaler()
                    df[f'{feature}_scaled'] = scaler.fit_transform(df[[feature]])
                    self.weather_scalers[feature] = scaler
                else:
                    if feature in self.weather_scalers:
                        df[f'{feature}_scaled'] = self.weather_scalers[feature].transform(df[[feature]])
        
        return df
    
    def create_sequences(self, df):
        """Create sequences for time series prediction"""
        print("Creating sequences...")
        
        # Select features for model input
        feature_cols = [
            # Temporal features
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_holiday', 'is_morning_rush', 'is_evening_rush', 'is_weekend',
            
            # Weather features (scaled)
            'temp_celsius_scaled', 'rain_1h_scaled', 'snow_1h_scaled', 
            'clouds_all_scaled', 'weather_severity_scaled',
            
            # Weather binary features
            'has_rain', 'has_snow', 'adverse_weather',
            'weather_clear', 'weather_clouds', 'weather_rain', 'weather_snow',
            
            # Temperature categories
            'temp_cold', 'temp_cool', 'temp_mild', 'temp_warm'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_names = available_features
        
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Prepare data
        feature_data = df[available_features].values
        target_data = df['traffic_volume_scaled'].values
        
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length - self.prediction_length + 1):
            # Input sequence
            X.append(feature_data[i:i + self.sequence_length])
            # Target sequence (next prediction_length time steps)
            y.append(target_data[i + self.sequence_length:i + self.sequence_length + self.prediction_length])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        print("Splitting data...")
        
        # Temporal split (earlier data for training)
        total_samples = len(X)
        test_start = int(total_samples * (1 - self.test_size))
        val_start = int(test_start * (1 - self.val_size))
        
        X_train = X[:val_start]
        y_train = y[:val_start]
        
        X_val = X[val_start:test_start]
        y_val = y[val_start:test_start]
        
        X_test = X[test_start:]
        y_test = y[test_start:]
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def preprocess(self, file_path):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        
        # Create features
        df = self.create_temporal_features(df)
        df = self.create_weather_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Scale features
        df = self.scale_features(df, fit=True)
        
        # Create sequences
        X, y = self.create_sequences(df)
        
        # Split data
        train_data, val_data, test_data = self.split_data(X, y)
        
        # Save preprocessing artifacts
        self.save_preprocessing_info()
        
        print("Preprocessing complete!")
        
        return train_data, val_data, test_data, df
    
    def save_preprocessing_info(self):
        """Save preprocessing information for later use"""
        preprocessing_info = {
            'sequence_length': self.sequence_length,
            'prediction_length': self.prediction_length,
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'scalers': {
                'traffic_scaler_min': float(self.traffic_scaler.data_min_[0]),
                'traffic_scaler_scale': float(self.traffic_scaler.scale_[0])
            }
        }
        
        # Save to JSON
        with open('../../data/processed/preprocessing_info.json', 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        print("Preprocessing info saved to preprocessing_info.json")

def main():
    """Main preprocessing function"""
    preprocessor = TrafficDataPreprocessor(
        sequence_length=12,  # 12 hours of data
        prediction_length=12,  # Predict next 12 hours
        test_size=0.1,
        val_size=0.2
    )
    
    # Run preprocessing
    train_data, val_data, test_data, processed_df = preprocessor.preprocess(
        '../../data/raw/Metro_Interstate_Traffic_Volume.csv'
    )
    
    # Save processed data
    np.save('../../data/processed/X_train.npy', train_data[0])
    np.save('../../data/processed/y_train.npy', train_data[1])
    np.save('../../data/processed/X_val.npy', val_data[0])
    np.save('../../data/processed/y_val.npy', val_data[1])
    np.save('../../data/processed/X_test.npy', test_data[0])
    np.save('../../data/processed/y_test.npy', test_data[1])
    
    print(f"Data saved:")
    print(f"Training: {train_data[0].shape} -> {train_data[1].shape}")
    print(f"Validation: {val_data[0].shape} -> {val_data[1].shape}")
    print(f"Test: {test_data[0].shape} -> {test_data[1].shape}")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    train_data, val_data, test_data = main()