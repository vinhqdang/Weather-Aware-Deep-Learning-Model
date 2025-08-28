"""
Weather Feature Encoder for Weather-Aware Traffic Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WeatherFeatureEncoder(nn.Module):
    """
    Weather Feature Encoder that transforms raw weather data into meaningful representations
    """
    
    def __init__(self, weather_dim=5, hidden_dim=64, output_dim=32, dropout=0.1):
        super(WeatherFeatureEncoder, self).__init__()
        
        self.weather_dim = weather_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Multi-layer encoder for weather features
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # Weather type classification head (auxiliary task)
        self.weather_classifier = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 4)  # 4 classes: clear, rain, fog, extreme
        )
        
        # Normalization layer
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, weather_features):
        """
        Forward pass through weather encoder
        
        Args:
            weather_features: [batch_size, seq_len, weather_dim] or [batch_size, weather_dim]
        
        Returns:
            weather_embedding: [batch_size, seq_len, output_dim] or [batch_size, output_dim]
            weather_type_logits: Weather classification logits for auxiliary loss
        """
        original_shape = weather_features.shape
        
        # Handle both batch and sequence dimensions
        if len(original_shape) == 3:
            batch_size, seq_len, _ = original_shape
            weather_features = weather_features.view(-1, self.weather_dim)
        
        # Encode weather features
        weather_embedding = self.weather_encoder(weather_features)
        weather_embedding = self.layer_norm(weather_embedding)
        
        # Weather type classification (auxiliary task)
        weather_type_logits = self.weather_classifier(weather_embedding)
        
        # Reshape back if needed
        if len(original_shape) == 3:
            weather_embedding = weather_embedding.view(batch_size, seq_len, self.output_dim)
            weather_type_logits = weather_type_logits.view(batch_size, seq_len, 4)
        
        return weather_embedding, weather_type_logits

class WeatherAttentionModule(nn.Module):
    """
    Weather Attention Module for spatial weather correlation
    """
    
    def __init__(self, weather_dim=32, num_heads=4, dropout=0.1):
        super(WeatherAttentionModule, self).__init__()
        
        self.weather_dim = weather_dim
        self.num_heads = num_heads
        self.head_dim = weather_dim // num_heads
        
        assert weather_dim % num_heads == 0, "weather_dim must be divisible by num_heads"
        
        # Multi-head attention for weather features
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=weather_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(weather_dim, weather_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(weather_dim * 2, weather_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(weather_dim)
        self.norm2 = nn.LayerNorm(weather_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, weather_embeddings, attention_mask=None):
        """
        Forward pass through weather attention
        
        Args:
            weather_embeddings: [batch_size, num_locations, weather_dim]
            attention_mask: Optional attention mask
        
        Returns:
            attended_weather: [batch_size, num_locations, weather_dim]
            attention_weights: [batch_size, num_heads, num_locations, num_locations]
        """
        # Self-attention
        attn_output, attention_weights = self.multihead_attn(
            weather_embeddings, weather_embeddings, weather_embeddings,
            attn_mask=attention_mask
        )
        
        # Residual connection and layer norm
        weather_embeddings = self.norm1(weather_embeddings + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(weather_embeddings)
        attended_weather = self.norm2(weather_embeddings + self.dropout(ffn_output))
        
        return attended_weather, attention_weights

class DynamicWeatherAdjacency(nn.Module):
    """
    Dynamic Weather Adjacency Matrix Generator
    Creates weather-adaptive spatial relationships
    """
    
    def __init__(self, weather_dim=32, hidden_dim=64, temperature=1.0):
        super(DynamicWeatherAdjacency, self).__init__()
        
        self.weather_dim = weather_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Weather similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(weather_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Edge weight temperature control
        self.temp_control = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, weather_embeddings, base_adjacency=None):
        """
        Generate dynamic adjacency matrix based on weather conditions
        
        Args:
            weather_embeddings: [batch_size, num_locations, weather_dim]
            base_adjacency: [num_locations, num_locations] base spatial adjacency
        
        Returns:
            dynamic_adjacency: [batch_size, num_locations, num_locations]
        """
        batch_size, num_locations, weather_dim = weather_embeddings.shape
        
        # Create pairwise weather feature combinations
        weather_i = weather_embeddings.unsqueeze(2).expand(-1, -1, num_locations, -1)
        weather_j = weather_embeddings.unsqueeze(1).expand(-1, num_locations, -1, -1)
        
        # Concatenate for similarity computation
        weather_pairs = torch.cat([weather_i, weather_j], dim=-1)
        weather_pairs = weather_pairs.view(-1, weather_dim * 2)
        
        # Compute weather similarity
        weather_similarity = self.similarity_net(weather_pairs)
        weather_similarity = weather_similarity.view(batch_size, num_locations, num_locations)
        
        # Apply temperature scaling
        weather_similarity = weather_similarity / self.temp_control
        
        # Combine with base adjacency if provided
        if base_adjacency is not None:
            base_adjacency = base_adjacency.unsqueeze(0).expand(batch_size, -1, -1)
            dynamic_adjacency = weather_similarity * base_adjacency
        else:
            dynamic_adjacency = weather_similarity
        
        # Ensure symmetry
        dynamic_adjacency = (dynamic_adjacency + dynamic_adjacency.transpose(-1, -2)) / 2
        
        # Add self-connections
        eye = torch.eye(num_locations, device=dynamic_adjacency.device)
        dynamic_adjacency = dynamic_adjacency + eye.unsqueeze(0)
        
        # Normalize
        row_sum = dynamic_adjacency.sum(dim=-1, keepdim=True)
        dynamic_adjacency = dynamic_adjacency / (row_sum + 1e-8)
        
        return dynamic_adjacency

class MultiScaleTemporalAttention(nn.Module):
    """
    Multi-Scale Temporal Attention with Weather Awareness
    """
    
    def __init__(self, input_dim, weather_dim=32, num_scales=3, num_heads=8, dropout=0.1):
        super(MultiScaleTemporalAttention, self).__init__()
        
        self.input_dim = input_dim
        self.weather_dim = weather_dim
        self.num_scales = num_scales
        self.num_heads = num_heads
        
        # Scale-specific attention heads
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_scales)
        ])
        
        # Weather-aware gating mechanism
        self.weather_gate = nn.Sequential(
            nn.Linear(weather_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # Output projection
        self.output_projection = nn.Linear(input_dim * num_scales, input_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, temporal_features, weather_context, attention_mask=None):
        """
        Multi-scale temporal attention with weather modulation
        
        Args:
            temporal_features: [batch_size, seq_len, input_dim]
            weather_context: [batch_size, weather_dim] or [batch_size, seq_len, weather_dim]
            attention_mask: Optional attention mask
        
        Returns:
            attended_features: [batch_size, seq_len, input_dim]
            attention_weights: List of attention weights for each scale
        """
        batch_size, seq_len, _ = temporal_features.shape
        
        # Handle weather context dimensions
        if len(weather_context.shape) == 2:
            weather_context = weather_context.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Weather-aware gating
        weather_gate = self.weather_gate(weather_context)
        gated_features = temporal_features * weather_gate
        
        # Multi-scale attention
        scale_outputs = []
        attention_weights = []
        
        for i, attention in enumerate(self.scale_attentions):
            # Apply scale-specific attention
            attn_output, attn_weights = attention(
                gated_features, gated_features, gated_features,
                attn_mask=attention_mask
            )
            
            # Weight by scale importance
            weighted_output = attn_output * self.scale_weights[i]
            scale_outputs.append(weighted_output)
            attention_weights.append(attn_weights)
        
        # Combine scales
        combined_output = torch.cat(scale_outputs, dim=-1)
        attended_features = self.output_projection(combined_output)
        
        # Residual connection and normalization
        attended_features = self.layer_norm(temporal_features + self.dropout(attended_features))
        
        return attended_features, attention_weights

def test_weather_encoder():
    """Test the weather encoder components"""
    
    # Test parameters
    batch_size = 32
    seq_len = 12
    weather_dim = 5
    num_locations = 1  # Single location for traffic data
    
    # Create test data
    weather_features = torch.randn(batch_size, seq_len, weather_dim)
    
    # Test Weather Feature Encoder
    print("Testing Weather Feature Encoder...")
    encoder = WeatherFeatureEncoder(weather_dim=weather_dim, output_dim=32)
    weather_embeddings, weather_logits = encoder(weather_features)
    print(f"Weather embeddings shape: {weather_embeddings.shape}")
    print(f"Weather logits shape: {weather_logits.shape}")
    
    # Test Weather Attention Module
    print("\nTesting Weather Attention Module...")
    weather_single = weather_embeddings[:, 0, :]  # [batch_size, weather_dim]
    weather_single = weather_single.unsqueeze(1)  # [batch_size, 1, weather_dim]
    
    attention_module = WeatherAttentionModule(weather_dim=32)
    attended_weather, attn_weights = attention_module(weather_single)
    print(f"Attended weather shape: {attended_weather.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test Dynamic Weather Adjacency
    print("\nTesting Dynamic Weather Adjacency...")
    adjacency_module = DynamicWeatherAdjacency(weather_dim=32)
    dynamic_adj = adjacency_module(weather_single)
    print(f"Dynamic adjacency shape: {dynamic_adj.shape}")
    
    # Test Multi-Scale Temporal Attention
    print("\nTesting Multi-Scale Temporal Attention...")
    temporal_features = torch.randn(batch_size, seq_len, 64)
    weather_context = weather_embeddings.mean(dim=1)  # [batch_size, weather_dim]
    
    temporal_attention = MultiScaleTemporalAttention(input_dim=64, weather_dim=32)
    attended_temporal, temporal_weights = temporal_attention(temporal_features, weather_context)
    print(f"Attended temporal features shape: {attended_temporal.shape}")
    print(f"Number of attention weight matrices: {len(temporal_weights)}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_weather_encoder()