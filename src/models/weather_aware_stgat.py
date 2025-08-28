"""
Weather-Aware Spatiotemporal Graph Attention Network (Weather-Aware STGAT)
Main model architecture combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from weather_encoder import (
    WeatherFeatureEncoder, 
    WeatherAttentionModule, 
    DynamicWeatherAdjacency, 
    MultiScaleTemporalAttention
)

class SpatialGraphAttention(nn.Module):
    """
    Spatial Graph Attention Layer with weather-adaptive adjacency
    """
    
    def __init__(self, input_dim, output_dim, num_heads=8, dropout=0.1):
        super(SpatialGraphAttention, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, node_features, adjacency_matrix):
        """
        Spatial graph attention forward pass
        
        Args:
            node_features: [batch_size, num_nodes, input_dim]
            adjacency_matrix: [batch_size, num_nodes, num_nodes]
        
        Returns:
            output: [batch_size, num_nodes, output_dim]
            attention_weights: [batch_size, num_heads, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Linear transformations
        Q = self.query(node_features)  # [batch_size, num_nodes, output_dim]
        K = self.key(node_features)
        V = self.value(node_features)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply adjacency mask (set non-connected nodes to -inf)
        adjacency_mask = adjacency_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores.masked_fill(adjacency_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, self.output_dim
        )
        output = self.output_proj(attended_values)
        
        # Residual connection (if dimensions match)
        if self.input_dim == self.output_dim:
            output = self.layer_norm(node_features + self.dropout(output))
        else:
            output = self.layer_norm(output)
        
        return output, attention_weights

class WeatherAwareSTGATLayer(nn.Module):
    """
    Single Weather-Aware STGAT Layer
    Combines spatial graph attention and temporal attention with weather awareness
    """
    
    def __init__(self, input_dim, hidden_dim, weather_dim=32, num_heads=8, dropout=0.1):
        super(WeatherAwareSTGATLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weather_dim = weather_dim
        
        # Spatial graph attention
        self.spatial_attention = SpatialGraphAttention(
            input_dim=input_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Multi-scale temporal attention
        self.temporal_attention = MultiScaleTemporalAttention(
            input_dim=hidden_dim,
            weather_dim=weather_dim,
            num_scales=3,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, weather_embeddings, adjacency_matrix):
        """
        Forward pass through Weather-Aware STGAT layer
        
        Args:
            node_features: [batch_size, seq_len, num_nodes, input_dim]
            weather_embeddings: [batch_size, seq_len, weather_dim]
            adjacency_matrix: [batch_size, num_nodes, num_nodes]
        
        Returns:
            output: [batch_size, seq_len, num_nodes, hidden_dim]
        """
        batch_size, seq_len, num_nodes, _ = node_features.shape
        
        # Process each time step
        spatial_outputs = []
        
        for t in range(seq_len):
            # Spatial attention at time t
            spatial_out, _ = self.spatial_attention(
                node_features[:, t, :, :],  # [batch_size, num_nodes, input_dim]
                adjacency_matrix
            )
            spatial_outputs.append(spatial_out)
        
        # Stack temporal dimension
        spatial_features = torch.stack(spatial_outputs, dim=1)  # [batch_size, seq_len, num_nodes, hidden_dim]
        
        # Temporal attention for each node
        temporal_outputs = []
        
        for n in range(num_nodes):
            # Temporal attention for node n
            temporal_out, _ = self.temporal_attention(
                spatial_features[:, :, n, :],  # [batch_size, seq_len, hidden_dim]
                weather_embeddings  # [batch_size, seq_len, weather_dim]
            )
            temporal_outputs.append(temporal_out)
        
        # Stack spatial dimension
        temporal_features = torch.stack(temporal_outputs, dim=2)  # [batch_size, seq_len, num_nodes, hidden_dim]
        
        # Fuse with weather information
        weather_expanded = weather_embeddings.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        fused_features = torch.cat([temporal_features, weather_expanded], dim=-1)
        
        # Apply feature fusion
        fused_shape = fused_features.shape
        fused_features = fused_features.view(-1, fused_shape[-1])
        output = self.feature_fusion(fused_features)
        output = output.view(*fused_shape[:-1], self.hidden_dim)
        
        # Residual connection if dimensions match
        if self.input_dim == self.hidden_dim:
            output = self.layer_norm(node_features + output)
        else:
            output = self.layer_norm(output)
        
        return output

class WeatherAwareSTGAT(nn.Module):
    """
    Complete Weather-Aware Spatiotemporal Graph Attention Network
    """
    
    def __init__(self, 
                 num_features=26,
                 weather_features=5,
                 hidden_dim=64,
                 weather_dim=32,
                 num_layers=3,
                 num_heads=8,
                 prediction_length=12,
                 dropout=0.1,
                 num_nodes=1):
        super(WeatherAwareSTGAT, self).__init__()
        
        self.num_features = num_features
        self.weather_features = weather_features
        self.hidden_dim = hidden_dim
        self.weather_dim = weather_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.num_nodes = num_nodes
        
        # Weather feature encoder
        self.weather_encoder = WeatherFeatureEncoder(
            weather_dim=weather_features,
            hidden_dim=64,
            output_dim=weather_dim,
            dropout=dropout
        )
        
        # Dynamic weather adjacency generator
        self.dynamic_adjacency = DynamicWeatherAdjacency(
            weather_dim=weather_dim,
            hidden_dim=64,
            temperature=1.0
        )
        
        # Input projection layer
        self.input_projection = nn.Linear(num_features, hidden_dim)
        
        # Weather-Aware STGAT layers
        self.stgat_layers = nn.ModuleList([
            WeatherAwareSTGATLayer(
                input_dim=hidden_dim if i > 0 else hidden_dim,
                hidden_dim=hidden_dim,
                weather_dim=weather_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        # Output prediction heads
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_length)
        )
        
        # Weather classification head (auxiliary task)
        self.weather_classification_weight = 0.1
        
    def extract_weather_features(self, features):
        """Extract weather features from input features"""
        # Assuming weather features are the first 5 features after temporal features
        # Based on preprocessing: temp_celsius_scaled, rain_1h_scaled, snow_1h_scaled, 
        # clouds_all_scaled, weather_severity_scaled are at indices 10-14
        weather_indices = [10, 11, 12, 13, 14]  # Adjust based on actual preprocessing
        return features[:, :, weather_indices]
    
    def forward(self, features, base_adjacency=None):
        """
        Forward pass through Weather-Aware STGAT
        
        Args:
            features: [batch_size, seq_len, num_features]
            base_adjacency: Optional base adjacency matrix
        
        Returns:
            predictions: [batch_size, prediction_length]
            weather_logits: Weather classification logits
            attention_weights: Dictionary of attention weights
        """
        batch_size, seq_len, _ = features.shape
        
        # Extract weather features
        weather_features = self.extract_weather_features(features)
        
        # Encode weather features
        weather_embeddings, weather_logits = self.weather_encoder(weather_features)
        
        # Generate dynamic adjacency matrix
        weather_context = weather_embeddings.mean(dim=1)  # [batch_size, weather_dim]
        weather_context = weather_context.unsqueeze(1)  # [batch_size, 1, weather_dim]
        
        if base_adjacency is None:
            # For single location, create identity matrix
            base_adjacency = torch.eye(self.num_nodes, device=features.device)
        
        dynamic_adj = self.dynamic_adjacency(weather_context, base_adjacency)
        
        # Project input features
        node_features = self.input_projection(features)
        node_features = node_features.unsqueeze(2)  # [batch_size, seq_len, 1, hidden_dim]
        
        # Pass through STGAT layers
        x = node_features
        attention_weights = {}
        
        for i, layer in enumerate(self.stgat_layers):
            x = layer(x, weather_embeddings, dynamic_adj)
            # Store attention weights if needed
        
        # Global temporal pooling
        x = x.mean(dim=1)  # [batch_size, num_nodes, hidden_dim]
        x = x.squeeze(1)   # [batch_size, hidden_dim] for single node
        
        # Generate predictions
        predictions = self.prediction_head(x)  # [batch_size, prediction_length]
        
        return predictions, weather_logits, attention_weights

def test_weather_aware_stgat():
    """Test the complete Weather-Aware STGAT model"""
    
    # Test parameters
    batch_size = 16
    seq_len = 12
    num_features = 26
    weather_features = 5
    prediction_length = 12
    
    # Create test data
    features = torch.randn(batch_size, seq_len, num_features)
    
    # Create model
    model = WeatherAwareSTGAT(
        num_features=num_features,
        weather_features=weather_features,
        hidden_dim=64,
        weather_dim=32,
        num_layers=3,
        num_heads=8,
        prediction_length=prediction_length,
        dropout=0.1,
        num_nodes=1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    print("Testing forward pass...")
    predictions, weather_logits, attention_weights = model(features)
    
    print(f"Input shape: {features.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Weather logits shape: {weather_logits.shape}")
    
    # Test loss computation
    target = torch.randn(batch_size, prediction_length)
    mse_loss = F.mse_loss(predictions, target)
    
    # Weather classification loss (auxiliary)
    weather_targets = torch.randint(0, 4, (batch_size, seq_len))
    weather_loss = F.cross_entropy(
        weather_logits.view(-1, 4), 
        weather_targets.view(-1)
    )
    
    total_loss = mse_loss + 0.1 * weather_loss
    
    print(f"MSE Loss: {mse_loss:.4f}")
    print(f"Weather Loss: {weather_loss:.4f}")
    print(f"Total Loss: {total_loss:.4f}")
    
    # Test backward pass
    print("Testing backward pass...")
    total_loss.backward()
    
    print("All tests passed!")
    
    return model

if __name__ == "__main__":
    model = test_weather_aware_stgat()