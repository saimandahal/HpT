#!/usr/bin/env python
# coding: utf-8

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

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training


from torch.utils.data import DataLoader, Dataset

import time

import model as model

import dataLoader as dataLoader

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 

device = torch.device(dev)



# 
file_path = './data/winds.dat'  

dataset = pd.read_csv(file_path, delimiter=',')


seq_length = 12  

test_start_index = 480
test_end_index = 2120

test_data = dataset[test_start_index:test_end_index]

train_data = pd.concat([dataset[:test_start_index], dataset[test_end_index:]])


data_values_train = train_data.values

data_values_test = test_data.values



dataset_train = dataLoader.TimeSeriesDataset(data_values_train, seq_length)

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle = True)

dataset_test = dataLoader.TimeSeriesDataset(data_values_test, seq_length)

dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle = False)

input_dim = 12

# 
model_dim = 512

num_heads = 8

num_layers = 12


dropout = 0.01


wind_model = model.WindModel(input_dim, model_dim, num_heads, num_layers, dropout).to(device)
wind_model_1 = model.WindModel(input_dim, model_dim, num_heads, num_layers, dropout).to(device)


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(wind_model.parameters(), lr=0.000001)
optimizer_1 = torch.optim.Adam(wind_model.parameters(), lr=0.000001)

    





# Implementation Main

def check_conditions_loss(previous_average, average_loss, total_iterations  , flag_loss):

    if previous_average[0] is not None and previous_average[1] is not None:
        calc1= (previous_average[0] - previous_average[1])
        calc2 = (previous_average[1] - average_loss)
        condition1 = ( calc1 / previous_average[0]) < 0.1
        condition2 = ( calc2/ previous_average[1]) < 0.1

        if condition1 and condition2 and (calc2 < calc1):
            flag_loss = True

    previous_average[0] = previous_average[1]
    previous_average[1] = average_loss

    return flag_loss,previous_average


def check_conditions_weight(previous_weight_1, total_iterations , previous_params ,flag_weight):
    current_q = None
    current_k = None
    current_v = None
    distance_qkv = 0

    all_q = []
    all_k = []
    all_v = []

    q_name = f'transformer_layers.0.multihead_attn.q_linear.weight'

    v_name = f'transformer_layers.0.multihead_attn.v_linear.weight'

    for name, param in wind_model.named_parameters():
        if name == q_name:
            all_q = torch.clone(param.data)

        if name == v_name:
            all_v = torch.clone(param.data)

    merged_qkv_weight = torch.cat((all_q, all_v), dim=1)

    current_params = merged_qkv_weight

    if previous_params is not None:
        distance_qkv = torch.norm(current_params - previous_params , p='fro')

        if previous_weight_1[0] is not None and previous_weight_1[1] is not None:
            condition1 = previous_weight_1[0] - previous_weight_1[1]
            condition2 = previous_weight_1[1] - distance_qkv.item()

            if condition1 /previous_weight_1[0]  < 0.01 and condition2 /previous_weight_1[1] < 0.01 and condition2 < condition1:
                flag_weight = True
                # print(f"Satisfied for weight at iteration {total_iterations}")

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

save1.append(f"input_embedding_1")
save1.append(f"input_embedding_2")


save1.append("decoder_layer_1")
save1.append("decoder_layer_2")
save1.append("decoder_layer_3")

# 
save1.append("dim_reduction_1")
save1.append("dim_reduction_2")
save1.append("dim_reduction_3")


config = LoraConfig(target_modules= target1,  modules_to_save= save1, r = 8 , lora_alpha = 64 , lora_dropout=0.1 )

wind_model_1 = copy.deepcopy(wind_model)
peft_model = get_peft_model(wind_model_1, config)

peft_model.print_trainable_parameters()


# with open('peft_model.txt', 'w') as file1:
#     file1.write(str(peft_model) + '\n')


# with open('param_wind.txt', 'w') as file1:
#     for name, param in wind_model.named_parameters():
#         file1.write(str(name) + '\n')

# Training Section


with open("wind_model_main.txt", "w") as file:
    file.write(str(wind_model))


status = True

index = 0

main_loss = []
lora_loss = []

wind_model.train()


num_epochs = 24

start = time.time()


# 

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

check_iterations_loss= 480
check_iterations_weight= 480

t_time = 0

for epoch in range(num_epochs):

    total_loss = 0

    index = 0
    
    total_batches = 0

    total_batches_main = 0

    total_loss = 0

    net_loss = 0
    total_batches = 0
    average_loss = 0

    loss_in_batch_1 = 0

    total_batches_lora = 0

    loss_stat_full = []
    loss_stat_lora = []
                    
    lossMain = 0
    lossLora = 0



    for batch_idx, (inputs, targets) in enumerate(dataloader_train):

        index += 1

        total_iterations += 1

        total_batches += 1

        inputs, targets = inputs.to(device), targets.to(device)

        if implement_main: #

            total_batches_main +=1

            optimizer.zero_grad()

            outputs = wind_model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            loss_in_batch = loss.item()
                
            loss_all.append(loss_in_batch)
        
        if implement_lora:

            total_batches_lora +=1

            if for_once:
                wind_model_1 = copy.deepcopy(wind_model)
                peft_model = get_peft_model(wind_model_1, config)
                for_once= False
                optimizer_1 = torch.optim.AdamW(peft_model.parameters(), lr= 0.0001, weight_decay = 0.0001)

            optimizer_1.zero_grad(peft_model.parameters())

            outputs_1 = peft_model(inputs)

            loss_1 = criterion(outputs_1, targets)

            loss_1.backward()

            optimizer_1.step()

            loss_in_batch_1 = loss_1.item()
            
            net_loss += loss_in_batch_1

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
                # print("Checking Conditions:")
                average_loss /= check_iterations_loss
                flag_loss,previous_average = check_conditions_loss(previous_average, average_loss, total_iterations , flag_loss )
                average_loss = 0

            if total_batches % check_iterations_weight == 0:

                flag_weight,previous_params,previous_weight_1= check_conditions_weight(previous_weight_1,total_iterations, previous_params,flag_weight )

        if flag_loss and flag_weight:

            t_time = time.time()
            print(f"Satisfied condition 1 at iteration {total_iterations} @ time {t_time-start} ")
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

            if(count % 480 == 0):

                result = t_test_check(loss_stat_full , loss_stat_lora)
                if result:
                    implement_lora = True
                    for_once = True
                    implement_main = False
                    count = 0
                    flag_loss = False

                    t_test = False

                    t_time = time.time()
        
                    print(f"Satisfied condition 1 at iteration {total_iterations} @ time {t_time-start} ")


                    end_time_t = time.time()
                    satisfied_iteration = total_iterations
                    stop_flag = True

                    t_time = end_time_t - start
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


    end_t = time.time()

    print(index)
    print(end_t-start)

            
    

end = time.time()
#
print("t-time", t_time)
# 
print("Time:" , end-start)




# Evaluation on Hybrid Model
peft_model.eval()



def test_model(wind_model, dataloader, criterion):
  
    total_loss = 0
  
    actual_values = []
  
    predicted_values = []
    
    with torch.no_grad():
  
        for batch_idx, (inputs, targets) in enumerate(dataloader):
  
            inputs, targets = inputs.to(device), targets.to(device)
  
            outputs = wind_model(inputs)


            targets_col13 = targets[:, 24]  # Assuming 0-based index
            outputs_col13 = outputs[:, 24]  # Assuming 0-based index
            
            # Reshape for criterion if necessary
            targets_col13 = targets_col13.view(-1, 1)
            outputs_col13 = outputs_col13.view(-1, 1)

            loss = criterion(outputs_col13, targets_col13)
            total_loss += loss.item()
            
            actual_values.append(targets_col13.cpu().numpy())
            predicted_values.append(outputs_col13.cpu().numpy())
  
            # loss = criterion(outputs, targets)
  
            # total_loss += loss.item()
            
            # actual_values.append(targets.cpu().numpy())

            # predicted_values.append(outputs.cpu().numpy())
    
    actual_values = np.concatenate(actual_values, axis=0)

    predicted_values = np.concatenate(predicted_values, axis=0)
    
    return total_loss / len(dataloader), actual_values, predicted_values

test_loss, actual_values, predicted_values = test_model(peft_model, dataloader_test, criterion)


print(f'Test Loss: {test_loss:.4f}')

actual_values = actual_values.flatten()  
predicted_values = predicted_values.flatten()

test_results = pd.DataFrame()

test_results['actual'] =pd.Series(actual_values)
test_results['predicted'] =pd.Series(predicted_values)


test_results.to_csv('/local/data/sdahal_p/wind/result/resultHy1.csv' , index = False)


data = pd.read_csv('/local/data/sdahal_p/wind/result/resultHy1.csv')

data['Squared_Error'] = (data['actual'] - data['predicted']) ** 2
rmse_per_row = np.sqrt(data['Squared_Error'])

average_rmse = rmse_per_row.mean()

rmse = pd.DataFrame()

rmse['RMSE Loss'] =pd.Series(average_rmse)
rmse['time'] =pd.Series((end-start))

rmse.to_csv('/local/data/sdahal_p/wind/result/rmseHy1.csv')