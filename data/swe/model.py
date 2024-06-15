
import torch
import torch.nn as nn


import random
import torch.nn.functional as F

import pandas as pd
import numpy as np

import hydroeval as he

import math

from scipy import stats

import copy

import time
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

import SDL as SDL

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 

device = torch.device(dev)

snotel_locations_info = pd.read_csv('Data/Snotel_Locations_Filtered_v3.csv')
snotel_locations_info['Southness'] = [ math.sin(value['Slope_tif1_x']) * math.cos(value['Aspect_tif_x']) for index,value in snotel_locations_info.iterrows()]


sp_dataLoader = SDL.SpatialDataLoader()

sp_dataLoader.cleanDataJunk()
sp_dataLoader.prepareInputsAndOutputs()

sp_training_input , sp_training_output = sp_dataLoader.getTrainingData()
sp_testing_data_1,sp_testing_data_2, sp_testing_data_3 , sp_testing_data_4,sp_testing_data_5 = sp_dataLoader.getTestingData()

sp_testing_input_1 ,sp_testing_output_1 = sp_testing_data_1
sp_testing_input_2 ,sp_testing_output_2 = sp_testing_data_2
sp_testing_input_3 ,sp_testing_output_3 = sp_testing_data_3
sp_testing_input_4 ,sp_testing_output_4 = sp_testing_data_4
sp_testing_input_5 ,sp_testing_output_5 = sp_testing_data_5

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

class SWETransformer(nn.Module):
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
          
        self.transformer_layers = nn.ModuleList([TransformerLayer(model_dim=self.model_dim, n_heads=16) for _ in range(12)])

        # Dimension Reduction.
        
        initial_dim  = self.model_dim
        
        self.dim_red_1 = torch.nn.Linear(initial_dim * 2 , int(initial_dim/2))
        self.dim_red_2 = torch.nn.Linear(int(initial_dim/2) , int(initial_dim/2))
        self.dim_red_3 = torch.nn.Linear(int(initial_dim/2) , int(initial_dim/4))
        self.dim_red_4 = torch.nn.Linear(int(initial_dim/4) , int(initial_dim/32))
        
        self.dim_red_dropout = torch.nn.Dropout(p = 0.05)
        
        # Final output layer for the model.
        
        self.decoder_layer_1 = torch.nn.Linear(self.seq_length * int(self.model_dim/32) ,self.seq_length * int(self.model_dim/128)  )    
        self.decoder_layer_2 = torch.nn.Linear(self.seq_length * int(self.model_dim/128), self.seq_length)
        
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
        
        encoder_inputs = torch.from_numpy(encoder_inputs).to(dtype= torch.float32,device=device)
                
        # Getting the configuration of the passed data.
        
        encoder_batch_size, encoder_sequence_length , encoder_input_dim = encoder_inputs.shape
                
        # Embedding the daily data passed to the model for the locations.
        
        
        embed_input_x = encoder_inputs.reshape(-1,self.encoder_input_dim)
        
        embed_input_x = self.input_embed_1(embed_input_x)
        embed_input_x = self.activation_gelu(embed_input_x)
        
        embed_input_x = self.input_embed_2(embed_input_x)
        embed_input_x = self.activation_gelu(embed_input_x)
        
        embed_input_x = embed_input_x.reshape(encoder_batch_size, encoder_sequence_length, self.model_dim)
        
        # Applying positional encoding.
        
        x = self.positional_encoding(embed_input_x)
        
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
        
        x = x.reshape(-1, encoder_sequence_length * int(self.model_dim/32))
        
        x= self.decoder_layer_1(x)
        x= self.activation_gelu(x)
        x = self.dropout_10(x)
        
        x = self.decoder_layer_2(x)
        x = self.activation_identity(x)
        
        x= x.reshape(encoder_batch_size , encoder_sequence_length , self.n_output_heads)

        return x
    

swe_model_sp = SWETransformer(encoder_input_dim = 19, model_dim = 512, n_output_heads = 1, seq_length = 323)
swe_model_sp_1 = SWETransformer(encoder_input_dim = 19, model_dim = 512, n_output_heads = 1, seq_length = 323)

swe_model_sp = swe_model_sp.to(device = device)
swe_model_sp_1 = swe_model_sp_1.to(device = device)

 
mean_squared_error_sp = nn.MSELoss()

# Optimizer
optimizer_sp = torch.optim.AdamW(swe_model_sp.parameters(), lr= 0.0001, weight_decay = 0.0001)
scheduler_sp = torch.optim.lr_scheduler.StepLR(optimizer_sp, step_size = 3 ,gamma = 0.6, last_epoch= -1, verbose=False)

optimizer_sp_1 = torch.optim.AdamW(swe_model_sp_1.parameters(), lr= 0.0001, weight_decay = 0.0001)
scheduler_sp_1 = torch.optim.lr_scheduler.StepLR(optimizer_sp_1, step_size = 3 ,gamma = 0.6, last_epoch= -1, verbose=False)


# Implementation Main

def check_conditions_loss(previous_average, average_loss, total_iterations  , flag_loss):
    # print("Chekcing loss and weight")
    if previous_average[0] is not None and previous_average[1] is not None:
        calc1= (previous_average[0] - previous_average[1])
        calc2 = (previous_average[1] - average_loss)
        condition1 = ( calc1 / previous_average[0]) < 0.1
        condition2 = ( calc2/ previous_average[1]) < 0.1

        if condition1 and condition2 and (calc2 < calc1):
            flag_loss = True
            # print(f"Satisfied loss at iteration {total_iterations}")

    # Update losses
    previous_average[0] = previous_average[1]
    previous_average[1] = average_loss
    # average_loss = 0

    return flag_loss,previous_average


def check_conditions_weight(previous_weight_1, total_iterations , previous_params ,flag_weight):
    current_q = None
    current_k = None
    current_v = None
    distance_qkv = 0

    all_q = []
    all_k = []
    all_v = []

    q_name = f'transformer_layers.11.multihead_attn.q_linear.weight'

    # k_name = f'transformer_layers.11.multihead_attn.k_linear.weight'
    v_name = f'transformer_layers.11.multihead_attn.v_linear.weight'

        # Iterate through parameters
    for name, param in swe_model_sp.named_parameters():
        if name == q_name:
            all_q = torch.clone(param.data)
        # if name == k_name:
        #     all_k.append(torch.clone(param.data))
        if name == v_name:
            all_v = torch.clone(param.data)

    merged_qkv_weight = torch.cat((all_q, all_v), dim=1)

    # if (current_k is not None) and (current_v is not None) and (current_q is not None):
    current_params = merged_qkv_weight

    if previous_params is not None:
        distance_qkv = torch.norm(current_params - previous_params , p='fro')
        # distance_qkv.item()
        print(distance_qkv.item())

        if previous_weight_1[0] is not None and previous_weight_1[1] is not None:
            condition1 = previous_weight_1[0] - previous_weight_1[1]
            condition2 = previous_weight_1[1] - distance_qkv.item()

            if condition1 /previous_weight_1[0]  < 0.01 and condition2 /previous_weight_1[1] < 0.01 and condition2 < condition1:
                flag_weight = True
                # print(f"Satisfied for weight at iteration {total_iterations}")

        # Update weights
        previous_weight_1[0] = previous_weight_1[1]
        previous_weight_1[1] = distance_qkv.item()
    previous_params = current_params
    

    return flag_weight , previous_params , previous_weight_1

def t_test_check(loss_stat_full, loss_stat_lora):
    t_statistic, p_value = stats.ttest_ind(loss_stat_full, loss_stat_lora)
    alpha = 0.01

    degrees_of_freedom = len(loss_stat_full) + len(loss_stat_lora) - 2

    critical_value = stats.t.ppf(1 - alpha/2, degrees_of_freedom)  
    
    if abs(t_statistic) < critical_value and p_value > alpha :
        # print("There is no significant difference.")
        return True
    else:
        # print("There is a significant difference.")
        return False

def TrainModelSP(train_inputs, train_outputs, epoch_number, total_iterations,  implement_main, implement_lora 
                  , check_conditions_flag , t_test, swe_model_sp_1 , for_once , peft_model , loss_all , optimizer_sp_1 , previous_params,previous_average,previous_weight_1 ):

    total_loss = 0
    net_loss = 0
    total_batches = 0
    average_loss = 0
    
    loss_in_batch_1 = 0

    total_batches_lora =0
    total_batches_main =0

    count = 0

    flag_loss = False
    flag_weight = False

    x= len(train_inputs)

    check_iterations_loss= 365
    check_iterations_weight= 365

    loss_stat_full = []
    loss_stat_lora = []
                    
    lossMain = 0
    lossLora = 0
    end_time = time.time()
    stop_flag = False

    for input_index , batch_input in enumerate(train_inputs):
                
        total_batches+=1
        total_iterations+=1


        batch_size , sequence_length , feature_dim = train_outputs[input_index].shape  

        if implement_main:  
            total_batches_main +=1

            optimizer_sp.zero_grad()
                
            output = swe_model_sp(batch_input)
                
            loss = mean_squared_error_sp(output, torch.from_numpy(train_outputs[input_index]).to(dtype=torch.float32,device=device))

            loss_in_batch = loss.item() * batch_input.shape[0]
                
            total_loss+=loss_in_batch

            loss.backward()

            optimizer_sp.step()

            loss_all.append(loss_in_batch)
        
        if implement_lora:
            total_batches_lora +=1

            if for_once:
                swe_model_sp_1 = copy.deepcopy(swe_model_sp)
                peft_model = get_peft_model(swe_model_sp_1, config)
                for_once= False
                optimizer_sp_1 = torch.optim.AdamW(peft_model.parameters(), lr= 0.0001, weight_decay = 0.0001)

            optimizer_sp_1.zero_grad(peft_model.parameters())
    
            output_1 = peft_model(batch_input)
    
            loss_1 = mean_squared_error_sp(output_1, torch.from_numpy(train_outputs[input_index]).to(dtype=torch.float32,device=device))

            loss_in_batch_1 = loss_1.item() * batch_input.shape[0]
            
            net_loss+=loss_in_batch_1
            
            loss_1.backward()
            
            optimizer_sp_1.step()

        # Loss lists
        if implement_lora and implement_main : 
            lora_list.append(loss_in_batch_1)
            main_list.append(loss_in_batch) 

        elif implement_lora and not implement_main:
            lora_list.append(loss_in_batch_1)
            main_list.append(0)

        elif implement_main and not implement_lora:
            lora_list.append(0)
            main_list.append(loss_in_batch)
             
        #  Conditions

        if check_conditions_flag:
            average_loss+= loss_in_batch
            if total_batches % check_iterations_loss == 0:
                print("Checking Conditions:")
                average_loss /= check_iterations_loss
                flag_loss,previous_average = check_conditions_loss(previous_average, average_loss, total_iterations , flag_loss )
                average_loss = 0

            if total_batches % check_iterations_weight == 0:

                flag_weight,previous_params,previous_weight_1= check_conditions_weight(previous_weight_1,total_iterations, previous_params,flag_weight )

        if flag_loss and flag_weight:
            print(f"Satisfied condition 1 at iteration {total_iterations}")
            check_conditions_flag = False
            flag_loss = False
            t_test = True

        else:
            flag_loss = False
            flag_weight = False
        
        if t_test:
            count+=1
            implement_lora = True

            if loss_in_batch_1 != 0:
                loss_stat_full.append(loss_in_batch)
                loss_stat_lora.append(loss_in_batch_1)

            if(count % 365 == 0):

                result = t_test_check(loss_stat_full , loss_stat_lora)
                if result:
                    implement_lora = True
                    for_once = True
                    implement_main = False
                    count = 0
                    flag_loss = False

                    t_test = False

                    print("t-test satisfied: Implementing Hybrid Model")
                    end_time = time.time()
                    stop_flag = True
                else:
                    implement_main = True
                    implement_lora = True
                    for_once = True

                    count = 0

                    loss_stat_full = []
                    loss_stat_lora = []

                    print("t-test not satisfied")
                    t_test = True

    if total_batches_main != 0:

        lossMain = total_loss / total_batches_main
        print("Main Training Loss:", lossMain)

    else:

        lossLora = net_loss / total_batches_lora
        print("LoRA training Loss: ", lossLora)

    return lossMain, lossLora, total_iterations,  check_conditions_flag, implement_main, implement_lora,for_once, peft_model, loss_all,optimizer_sp_1, t_test,end_time, stop_flag
 
# flag

for_once = True

t_test = False

check_conditions_flag = True

implement_lora= False
implement_main = True

# Initialization
total_iterations=0
loss_all = []

loss_main=[]
loss_lora=[]

previous_params = None

main_list = []
lora_list = []

previous_average = [None, None]
previous_weight_1 = [None, None]

distance_average=0
total_weight=0

current_q = None
current_k = None
current_v = None

# Hybrid Parameters
target1=[]
save1 = []

for layer in range(12):
    target1.append(f"transformer_layers.{layer}.multihead_attn.q_linear")
    target1.append(f"transformer_layers.{layer}.multihead_attn.v_linear")

save1.append(f"input_embed_1")
save1.append(f"input_embed_2")

save1.append("decoder_layer_1")
save1.append("decoder_layer_2")
save1.append("dim_red_1")
save1.append("dim_red_2")
save1.append("dim_red_3")
save1.append("dim_red_4")

config = LoraConfig(target_modules= target1,  modules_to_save= save1, r = 8 , lora_alpha = 64 , lora_dropout=0.1 )

swe_model_sp_1 = copy.deepcopy(swe_model_sp)
peft_model = get_peft_model(swe_model_sp_1, config)

peft_model.print_trainable_parameters()

# Training Section
swe_model_sp.train(True)
start_time = time.time()

status = True

index = 0

main_loss = []
lora_loss = []
while status:

    temp_holder_sp = list(zip(sp_training_input, sp_training_output))
    random.shuffle(temp_holder_sp)

    epoch_number = index+ 1
       
    train_input_batches_sp, train_output_batches_sp = zip(*temp_holder_sp)

    main1,lora1, total_iterations , check_conditions_flag, implement_main, implement_lora, for_once, peft_model , loss_all, optimizer_sp_1, t_test, end_time, stop_flag= TrainModelSP(train_input_batches_sp, train_output_batches_sp,epoch_number , total_iterations,
                            implement_main, implement_lora  , check_conditions_flag , t_test, swe_model_sp_1 , for_once,peft_model, loss_all ,optimizer_sp_1 , previous_params  , previous_average, previous_weight_1)
    scheduler_sp.step()

    index = index + 1

    if index > 23:
        status = False

    main_loss.append(main1)

    status_loss = False
    status_validation = False
    
    lora_loss.append(lora1)

end_time = time.time()
print("Time Elapsed", end_time - start_time)
time1 = end_time - start_time
loss_df1 = pd.DataFrame()

loss_df1['Train Loss'] =pd.Series(main_loss)
loss_df1['Lora Loss'] =pd.Series(lora_loss)



loss_df1['time'] =pd.Series(time1)

loss_df1.to_csv('/results/Train_Hybrid12.csv', index= False)

peft_model.eval()


def TestModelSP(test_inputs, test_outputs):

    losses = []
    
    mse_sp = nn.MSELoss()

    outputs = []
    outputs_loss = []
    actual_outputs = []
    
    with torch.no_grad():
        for input_index , batch_input in enumerate(test_inputs):
            
            output = peft_model(batch_input)
                        
            loss = mse_sp(output, torch.from_numpy(test_outputs[input_index]).to(dtype=torch.float32,device=device))
            
            outputs.append(output.detach().cpu().numpy())
            outputs_loss.append(loss.item())
            
            nse = he.evaluator(he.nse, output.detach().cpu().numpy().squeeze(), test_outputs[input_index].squeeze())

            actual_outputs.append(test_outputs[input_index])
            
            losses.append(loss.item())
    
    return (np.array(outputs), np.array(actual_outputs), np.array(outputs_loss))


test_outputs_1_sp, test_outputs_actual_1_sp, test_losses_1_sp = TestModelSP(sp_testing_input_1,sp_testing_output_1)
test_outputs_2_sp, test_outputs_actual_2_sp, test_losses_2_sp = TestModelSP(sp_testing_input_2,sp_testing_output_2)
test_outputs_3_sp, test_outputs_actual_3_sp, test_losses_3_sp = TestModelSP(sp_testing_input_3,sp_testing_output_3)
test_outputs_4_sp, test_outputs_actual_4_sp, test_losses_4_sp = TestModelSP(sp_testing_input_4,sp_testing_output_4)
test_outputs_5_sp, test_outputs_actual_5_sp, test_losses_5_sp = TestModelSP(sp_testing_input_5,sp_testing_output_5)

def getDFForOutputs(predicted_output,actual_outputs , loss_T):
    actual_outputs_df = pd.DataFrame()
    predicted_outputs_df = pd.DataFrame()

    station_index = snotel_locations_info[snotel_locations_info['Station Name'] == "Base Camp"].index[0]

    tp_output = predicted_output[:, 0, station_index, 0]
    target_output = actual_outputs[:, 0, station_index, 0]

    df = pd.DataFrame({'Actual': target_output, 'Predicted': tp_output , 'Loss': loss_T})

    return df

test_1_actual_df = getDFForOutputs(test_outputs_1_sp,test_outputs_actual_1_sp ,test_losses_1_sp )
test_2_actual_df = getDFForOutputs(test_outputs_2_sp,test_outputs_actual_2_sp , test_losses_2_sp)
test_3_actual_df = getDFForOutputs(test_outputs_3_sp,test_outputs_actual_3_sp , test_losses_3_sp)
test_4_actual_df = getDFForOutputs(test_outputs_4_sp,test_outputs_actual_4_sp , test_losses_4_sp)
test_5_actual_df = getDFForOutputs(test_outputs_5_sp,test_outputs_actual_5_sp , test_losses_5_sp)

test_1_actual_df.to_csv('/results/2014_test_results.csv')
test_2_actual_df.to_csv('/results/2006_test_results.csv')
test_3_actual_df.to_csv('/results/2017_test_results.csv')
test_4_actual_df.to_csv('/results/2016_test_results.csv')
test_5_actual_df.to_csv('/results/2007_test_results.csv')


def merge_csv_files(csv_paths):
    dfs = []
    
    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df

csv_paths = [
    '/results/2006_test_results.csv',
    '/results/2007_test_results.csv',
    '/results/2014_test_results.csv',
    '/results/2016_test_results.csv',
    '/results/2017_test_results.csv',
    ]

merged_df = merge_csv_files(csv_paths)
merged_df.to_csv('/results/test_Hybrid12_combined.csv')