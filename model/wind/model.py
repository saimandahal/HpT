
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torch.nn as nn

import math

import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Positional Encoding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).to(device=device)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device=device)

        pe = torch.zeros(max_len, 1, d_model).to(device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0)]

        return self.dropout(x)

# Multi-Head Attention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim, n_heads):

        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim

        self.n_heads = n_heads

        self.head_dim = model_dim // n_heads

        assert self.head_dim * n_heads == model_dim, "model_dim must be divisible by n_heads"

        self.q_linear = nn.Linear(model_dim, model_dim)

        self.k_linear = nn.Linear(model_dim, model_dim)

        self.v_linear = nn.Linear(model_dim, model_dim)

        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):

        batch_size = query.size(0)

        query = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.model_dim)

        return self.out(output)


# Feed Forward Network

class FeedForward(nn.Module):

    def __init__(self, model_dim, ff_dim):

        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Transformer Layer

class TransformerLayer(nn.Module):

    def __init__(self, model_dim, n_heads):

        super(TransformerLayer, self).__init__()

        self.multihead_attn = MultiHeadAttention(model_dim, n_heads)

        self.feed_forward = FeedForward(model_dim, model_dim * 2)

        self.layer_norm1 = nn.LayerNorm(model_dim)

        self.layer_norm2 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        x = self.layer_norm1(x + self.dropout(self.multihead_attn(x, x, x)))

        return self.layer_norm2(x + self.dropout(self.feed_forward(x)))


# Transformer Model

class WindModel(nn.Module):
    
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        
        super(WindModel, self).__init__()

        initial_dim  = model_dim


        
        self.input_embedding_1 = nn.Linear(input_dim, 256)
        
        self.input_embedding_2 = nn.Linear(256, initial_dim)

        self.positional_encoding = PositionalEncoding(initial_dim, dropout)
        
        self.transformer_layers = nn.ModuleList([TransformerLayer(initial_dim, num_heads) for _ in range(num_layers)])

        self.dropout_1 = torch.nn.Dropout(p = 0.1)

        self.dim_reduction_1 = torch.nn.Linear(initial_dim, int(initial_dim/4))
        
        self.dim_reduction_2 = torch.nn.Linear(int(initial_dim/4) , int(initial_dim/8))

        self.activation_relu = torch.nn.ReLU()

        # 

        self.decoder_layer_1 = torch.nn.Linear(int(initial_dim/8) *  57 , initial_dim )

        self.decoder_layer_2 = torch.nn.Linear(initial_dim , int(initial_dim/4))    

        self.decoder_layer_3 = torch.nn.Linear( int(initial_dim/4) , 57)


    def forward(self, src):


        src = src.permute(0, 2, 1)

        src = src.reshape(-1,12)

        x = self.input_embedding_1(src)

        x = self.input_embedding_2(x)

        src = self.positional_encoding(x)



        for layer in self.transformer_layers:
            x = layer(x)

        x = torch.cat((x , src),1)

        # x = x[:, -1, :]

        x = self.dim_reduction_1(x) 
        
        x= self.dropout_1(x)
        
        x = self.dim_reduction_2(x)

        x = self.activation_relu(x)
        
        x= self.dropout_1(x)

        # 
        
        x = x.reshape(-1, 57 * int(64))

        x = self.decoder_layer_1(x)

        x = self.activation_relu(x)

        x = self.decoder_layer_2(x)

        x = self.activation_relu(x)

        x = self.decoder_layer_3(x)

        return x

