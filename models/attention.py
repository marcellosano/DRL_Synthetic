import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for spatial awareness of hazards"""

    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Transform and reshape for multi-head attention
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out(context)

        return output, attention


class SpatialAttentionLayer(nn.Module):
    """Spatial attention layer for hazard awareness"""

    def __init__(self, input_dim=100, hidden_dim=256, num_heads=8):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # Project input
        x = self.input_projection(x)

        # Apply attention
        attended, weights = self.attention(x, x, x, mask)

        # Residual connection and layer norm
        x = self.layer_norm(x + attended)

        return x, weights