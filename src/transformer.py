import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Add dropout after embedding
        self.embed_dropout = nn.Dropout(dropout)
        
        # Create transformer encoder with dropout
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Add dropout before final linear layer
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
        # Initialize parameters with Xavier uniform
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src):
        # Create padding mask
        padding_mask = (src == 0)
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        # Apply embedding dropout
        x = self.embed_dropout(x)
        
        # Transformer layers
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Apply output dropout
        x = self.output_dropout(x)
        
        # Final linear layer
        output = self.output_layer(x)
        
        return output
