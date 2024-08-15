
import torch
import torch.nn as nn

import random

import pandas as pd
import numpy as np

import hydroeval as he

import torch.nn.functional as F

import os

from scipy import stats

import math

import time

import DL as DL

# Device
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 

device = torch.device(dev)

# File import
folder_path = '/data/'

# Dataloader

files = os.listdir(folder_path)
csv_files = [file for file in files if file.endswith('.csv')]

csv_path = [os.path.join(folder_path, csv_file) for csv_file in csv_files]

weather_dataLoader = DL.WeatherLoader(csv_path)

weather_dataLoader.cleanData()

# Testing data
w_testing_input,w_testing_output=weather_dataLoader.getTestingData()

# Training Data
w_training_input , w_training_output = weather_dataLoader.getTrainingData()


# Positional Encoding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).to(device=device)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device = device)
        pe = torch.zeros(max_len, 1, d_model).to(device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term).to(device=device)
        pe[:, 0, 1::2] = torch.cos(position * div_term).to(device=device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        batch_size, seq_len , embedding_dim = x.shape
        
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        
        seq_len , batch_size, embedding_dim = x.shape
        
        return x.reshape(batch_size,seq_len,embedding_dim)

class TransformerLayer(nn.Module):
   def __init__(self, model_dim, n_heads):

        super(TransformerLayer, self).__init__()

        self.multihead_attn = MultiHeadAttention(model_dim, n_heads)

        self.feed_forward = FeedForward(model_dim, model_dim * 4)

        self.layer_norm1 = nn.LayerNorm(model_dim)

        self.layer_norm2 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(0.1)
   def forward(self, x):
        
        # Multi-head attention
        residual = x
        
        x = self.layer_norm1(x + self.dropout(self.multihead_attn(x, x, x)))

        # Feed-forward
        x = self.layer_norm2(x + self.dropout(self.feed_forward(x)))

        return x

class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim):
        
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(model_dim, ff_dim)
        
        self.fc2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_heads):
        
        super(MultiHeadAttention, self).__init__()
        
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        
        assert self.head_dim * n_heads == model_dim, "Model"
        
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        
        batch_size = query.size(0)
        # Linear projections
        
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention calculation
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.model_dim)

        # Final linear layer
        output = self.out(output)
        return output 


class WeatherTrans(nn.Module):
    
    def __init__(self, encoder_input_dim = 20, model_dim = 64, n_output_heads = 1,window = 3,
                  seq_length = 10):
        super().__init__()
        
        # Storing the passed argument on the class definition.
        
        self.model_dim = model_dim
        self.encoder_input_dim = encoder_input_dim
        self.n_output_heads = n_output_heads
        self.seq_length = seq_length
        
        # Linear Layers for the input.
        
        self.input_embed_1 = torch.nn.Linear(self.encoder_input_dim , int(self.model_dim/2))
        self.input_embed_2 = torch.nn.Linear(int(self.model_dim/2) , self.model_dim)

        self.input_dropout = torch.nn.Dropout(p = 0.10)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(self.model_dim,0.2)
           
        # Transformer model definition.
        self.transformer_layers = nn.ModuleList([TransformerLayer(model_dim=self.model_dim, n_heads=16) for _ in range(12)])
        
        # Dimension Reduction.
        
        initial_dim  = self.model_dim
        
        self.dim_red_1 = torch.nn.Linear(initial_dim * 2 , int(initial_dim/2))
        self.dim_red_2 = torch.nn.Linear(int(initial_dim/2) , int(initial_dim/2))
        self.dim_red_3 = torch.nn.Linear(int(initial_dim/2) , int(initial_dim/4))
        self.dim_red_4 = torch.nn.Linear(int(initial_dim/4) , int(initial_dim/8))
        
        self.dim_red_dropout = torch.nn.Dropout(p = 0.05)
        
        # Final output layer for the model.
        
        self.decoder_layer_1 = torch.nn.Linear(self.seq_length * int(self.model_dim/8) ,self.seq_length * int(self.model_dim/16))    
        self.decoder_layer_2 = torch.nn.Linear(self.seq_length * int(self.model_dim/16), self.seq_length)
        
        # Activation Functions
        
        self.activation_relu = torch.nn.ReLU()
        self.activation_identity = torch.nn.Identity()
        self.activation_gelu = torch.nn.GELU()
        self.activation_tanh = torch.nn.Tanh()
        self.activation_sigmoid = torch.nn.Sigmoid()
        
        # Dropout Functions 
        
        self.dropout_5 = torch.nn.Dropout(p = 0.05)
        self.dropout_10 = torch.nn.Dropout(p = 0.10)
        self.dropout_15 = torch.nn.Dropout(p = 0.15)
        self.dropout_20 = torch.nn.Dropout(p = 0.20)
        
        
    def forward(self,encoder_inputs):
        
        # Converting to the torch array.
        encoder_inputs = encoder_inputs.astype(np.float32)

        encoder_inputs = torch.from_numpy(encoder_inputs).to(dtype= torch.float32,device=device)
                
        # Getting the configuration of the passed data.

        encoder_batch_size=1
        
        encoder_batch_size,encoder_sequence_length , encoder_input_dim = encoder_inputs.shape
        # Embedding the daily data passed to the model for the locations.

        
        embed_input_x = encoder_inputs.reshape(-1,self.encoder_input_dim)
        
        embed_input_x = self.input_embed_1(embed_input_x)
        embed_input_x = self.activation_gelu(embed_input_x)
        
        embed_input_x = self.input_embed_2(embed_input_x)
        embed_input_x = self.activation_gelu(embed_input_x)
        
        embed_input_x = embed_input_x.reshape(encoder_batch_size, encoder_sequence_length, self.model_dim)
        
        # Applying positional encoding.
        
        x = self.positional_encoding(embed_input_x)
        
        # Applying the transformer layer.

        # x = self.transformers_1(x)

        for layer in self.transformer_layers:
            x = layer(x)
        
        x = x.reshape(-1, self.model_dim)
        embed_input_x = embed_input_x.reshape(-1,self.model_dim)
        
        x = torch.cat((x , embed_input_x),1)
        
        # Dim reduction layer.
        
        x = self.dim_red_1(x) 
        x= self.dropout_20(x)
        
        x = self.dim_red_2(x)
        x = self.activation_relu(x)
        x= self.dropout_20(x)

        x = self.dim_red_3(x)
        x= self.dropout_20(x)
        
        x= self.dim_red_4(x)
        x = self.activation_relu(x)
        x= self.dropout_20(x)
        
        
        
        # Final layer for the output.
        
        x = x.reshape(-1, encoder_sequence_length * int(self.model_dim/8))
        
        x= self.decoder_layer_1(x)
        x= self.activation_gelu(x)
        x = self.dropout_10(x)
        
        x = self.decoder_layer_2(x)
        x = self.activation_identity(x)
        
        x= x.reshape(encoder_batch_size , encoder_sequence_length , self.n_output_heads)
        
        return x

    
# Model
weather_model = WeatherTrans(encoder_input_dim = 13, model_dim = 512, n_output_heads = 1, seq_length = 29)

weather_model = weather_model.to(device = device)

mean_squared_error_weather = nn.MSELoss()

# Optimizer
optimizer_weather = torch.optim.AdamW(weather_model.parameters(), lr= 0.0001, weight_decay = 0.0001)
scheduler_weather = torch.optim.lr_scheduler.StepLR(optimizer_weather, step_size = 3 ,gamma = 0.6, last_epoch= -1, verbose=False)


# Implementation Main

def TrainModelWeather(train_inputs, train_outputs, epoch_number, total_iterations):

    total_loss = 0
    total_batches = 0
    average_loss = 0

    x= len(train_inputs)

    lossMain = 0

    for input_index , batch_input in enumerate(train_inputs):
                
        total_batches+=1
        total_iterations+=1

        batch_size , sequence_length , feature_dim = train_outputs[input_index].shape  

        optimizer_weather.zero_grad()
            
        output = weather_model(batch_input)
            
        loss = mean_squared_error_weather(output, torch.from_numpy(train_outputs[input_index]).to(dtype=torch.float32,device=device))

        loss_in_batch = loss.item() * batch_input.shape[0]
            
        total_loss+=loss_in_batch

        loss.backward()

        optimizer_weather.step()

    lossMain = total_loss / total_batches
    print("Average Training Loss:", lossMain)

    return total_iterations
 
# Initialization
total_iterations=0

# Training Section
weather_model.train(True)

start_time = time.time()

status = True

index = 0

while status:

    temp_holder_weather = list(zip(w_training_input, w_training_output))
    random.shuffle(temp_holder_weather)

    epoch_number = index+ 1
       
    train_input_batches_w, train_output_batches_w = zip(*temp_holder_weather)

    total_iterations= TrainModelWeather(train_input_batches_w, train_output_batches_w,epoch_number , total_iterations)
    scheduler_weather.step()

    index = index + 1

    if index > 21:
        status = False

end_time = time.time()
print("Time Elapsed", end_time - start_time)

# Testing
weather_model.eval()

def TestModelWeather(test_inputs, test_outputs):
    loss_value = 0
    losses = []
    
    mse_weather = nn.MSELoss()

    outputs = []
    outputs_loss = []
    actual_outputs = []
    
    
    with torch.no_grad():
        for input_index , batch_input in enumerate(test_inputs):
            
            output = weather_model(batch_input)
                        
            loss = mse_weather(output, torch.from_numpy(test_outputs[input_index]).to(dtype=torch.float32,device=device))
            
            outputs.append(output.detach().cpu().numpy())
            outputs_loss.append(loss.item())
            

            actual_outputs.append(test_outputs[input_index])
            
            losses.append(loss.item())
    
    return (np.array(outputs), np.array(actual_outputs), np.array(outputs_loss))


test_outputs_1_w, test_outputs_actual_1_w, test_losses_1_w = TestModelWeather(w_testing_input,w_testing_output)

# DataFrame preparation
def getDFForOutputs(predicted_output, actual_outputs, csv_paths):
    actual_outputs_df = pd.DataFrame()
    predicted_outputs_df = pd.DataFrame()

    station_index = None
    for index, path in enumerate(csv_paths):
        if path == '/data/AWN_60_100017.csv':
            station_index = index
            break

    tp_output = predicted_output[:, 0, station_index, 0]
    target_output = actual_outputs[:, 0, station_index, 0]

    df = pd.DataFrame({'Actual': target_output, 'Predicted': tp_output})

    return df


test_1_actual_df = getDFForOutputs(test_outputs_1_w,test_outputs_actual_1_w, csv_path)
# CSV file
test_1_actual_df.to_csv('/results/test_Layers.csv')
