import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int
    pad_token_id: int
    seq_length: int
    hidden_size: int
    head_size: int
    num_heads: int
    num_layers: int
    dropout_rate: float
    intermediate_size: int

class TransformerComponent(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        
        # Weights
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta  = nn.Parameter(torch.zeros(hidden_size))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def normalize(self, inputs, outputs):
        """Apply residual connections and layer normalization."""
        x = inputs + outputs # (batch size, input length, hidden size)
        
        # Normalize
        mean = torch.mean(x, dim = 1, keepdim = True)
        std  = torch.std(x, dim = 1, keepdim = True)
        
        normalized = (x - mean) / std
        
        # Scale & Shift
        normalized = normalized * self.gamma + self.beta
        
        return normalized
    
    def __call__(self, *args):
        # Compute the outputs
        outputs = self.forward(*args)
        
        # Apply Dropout
        outputs = self.dropout(outputs) # Dropout is applied after each component output (self-attention & ffn)
        
        # Add & Normalize
        normalized = self.normalize(args[0], outputs)
        
        return normalized

class PositionalEncoding(nn.Module):
    def __init__(self, max_pos, hidden_size):
        super().__init__()
        
        # Compute Positional Encodings
        positons = torch.arange(max_pos).unsqueeze(1)
        dimentions = torch.arange(hidden_size).unsqueeze(0)
        
        angle_rates = 1 / (10000 ** (dimentions / hidden_size))
        angles = positons * angle_rates
        
        pe = torch.zeros(max_pos, hidden_size)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        
        self.embedding = nn.Embedding.from_pretrained(pe)
    
    def forward(self, inputs):
        positions = torch.arange(
            inputs.shape[1], device = inputs.device
        )
        
        encodings = self.embedding(positions)
        
        return inputs + encodings.unsqueeze(0)

class MultiHeadAttention(TransformerComponent):
    def __init__(self, hidden_size, head_size, num_heads, dropout_rate):
        super().__init__(hidden_size, dropout_rate)

        # Params
        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = head_size * num_heads
        
        # Weights
        self.wq = nn.Linear(hidden_size, self.output_size)
        self.wk = nn.Linear(hidden_size, self.output_size)
        self.wv = nn.Linear(hidden_size, self.output_size)
        self.wo = nn.Linear(self.output_size, hidden_size)
    
    def forward(self, inputs, attention_mask):
        batch, length, _ = inputs.shape
        
        # Linear Projections
        reshaped = (batch, length, self.num_heads, self.head_size)
        query = self.wq(inputs).reshape(reshaped).permute(0, 2, 1, 3)
        key   = self.wk(inputs).reshape(reshaped).permute(0, 2, 1, 3)
        value = self.wv(inputs).reshape(reshaped).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = (query @ key.transpose(-2, -1)) / (self.head_size ** 0.5)
        
        # Masking
        key_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, length, -1)
        masked_scores = scores.masked_fill((key_mask == 0), float("-inf"))
        
        # Compute attention values
        weights = F.softmax(masked_scores, dim = -1)

        query_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
        masked_weights = weights * query_mask
    
        attention_per_head = (masked_weights @ value).permute(0, 2, 1, 3)
        
        # Concatenation
        combined = attention_per_head.reshape(batch, length, self.output_size)
        
        # Final Output Projection
        outputs = self.wo(combined)
        
        return outputs
    
class FFN(TransformerComponent):
    def __init__(self, hidden_size, intermediate_size, dropout_rate):
        super().__init__(hidden_size, dropout_rate)
        
        # Weights
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )
    
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            config.hidden_size, config.head_size, config.num_heads, config.dropout_rate
        )
        
        self.ffn = FFN(
            config.hidden_size, config.intermediate_size, config.dropout_rate
        )
    
    def forward(self, inputs, attention_mask):
        attention_out = self.attention(inputs, attention_mask)
        
        ffn_out = self.ffn(attention_out)
        
        return ffn_out

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            config.seq_length, config.hidden_size
        )
        
        # Stacked Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, inputs, attention_mask):
        y = self.embedding(inputs)
        
        y = self.positional_encoding(y)
        
        y = self.dropout(y) # Dropout is applied on embeddings + positional encodings
        
        for layer in self.layers:
            y = layer(y, attention_mask)
        
        return y