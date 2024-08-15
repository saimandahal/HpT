import os 

import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torch.nn as nn

import csv

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import random

from sklearn.metrics import mean_absolute_error, mean_squared_error

import math

import time


from scipy import stats

import copy

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training


from torch.utils.data import DataLoader, Dataset


import torch.nn.functional as F

import torch.optim as optim

import dataLoader as dataLoader

import model as modelC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Data Preparation
data = pd.read_csv('./data/covid.csv')

data = data[data['Date'] <= 230]

data_test = data[data['Date'] >= 209]


counties = list(pd.unique(data['GISJOIN']))

counties_sample = random.sample(counties, int(np.floor(len(counties) * 0.1)))

data_train = data[data['Date'] < 209]

variables = ['deaths','confirmed_cases','foot_traffic','Race1','Race2','Race3','Race4','FHH','HHS1','HHS2','HHS3','SE1','PL','MHHI','MNR','MGR','MHV','HI','MHI']



feature_vectors_train = dataLoader.create_feature_vector_train(counties_sample, data_train, 3, variables)

feature_vectors_test = dataLoader.create_feature_vector_test(counties_sample, data_test, 3, variables)



# Model
covid_model = modelC.CovidModel(input_dim=1, model_dim=512, num_heads=8, num_layers=6, dropout=0.001).to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(covid_model.parameters(), lr=0.00001)



# Initial convergence test
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

    for name, param in covid_model.named_parameters():
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

# T-test

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

save1.append("dim_red_1")
save1.append("dim_red_2")


config = LoraConfig(target_modules= target1,  modules_to_save= save1, r = 8 , lora_alpha = 64 , lora_dropout=0.1 )

covid_model_1 = copy.deepcopy(covid_model)
peft_model = get_peft_model(covid_model_1, config)

peft_model.print_trainable_parameters()



status = True

index = 0

main_loss = []
lora_loss = []


# Training
covid_model.train()

start = time.time()

num_epochs = 20


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

check_iterations_loss= 720
check_iterations_weight= 720


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

    epoch_loss = 0

    for batch in feature_vectors_train:

        index += 1

        total_iterations += 1

        total_batches += 1
        
        inputs, targets = batch

        inputs, targets = inputs.to(device), targets.to(device)


        if implement_main: #

            total_batches_main +=1

            optimizer.zero_grad()

            outputs = covid_model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            loss_in_batch = loss.item()
                
            loss_all.append(loss_in_batch)
        
        
        if implement_lora:

            total_batches_lora +=1

            if for_once:
                covid_model_1 = copy.deepcopy(covid_model)
                peft_model = get_peft_model(covid_model_1, config)
                for_once= False
                optimizer_1 = torch.optim.AdamW(peft_model.parameters(), lr= 0.00001, weight_decay = 0.001)

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

            if(count % 720 == 0):

                result = t_test_check(loss_stat_full , loss_stat_lora)

                if result:

                    implement_lora = True
                    for_once = True
                    implement_main = False
                    count = 0
                    flag_loss = False

                    t_test = False

                    print("t-test satisfied: Implementing Hybrid Model", total_iterations)
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


end = time.time()

t_time = end_time_t - start
print(f'Training completed in {end - start:.2f} seconds.')

# Testing

peft_model.eval()


def evaluate_model(model, feature_vectors_test):

    true_values = []
    predicted_values = []

    with torch.no_grad():

        for batch in feature_vectors_test:

            inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            
            true_values.append(targets.cpu().numpy())

            predicted_values.append(outputs.cpu().numpy())

    true_values = np.concatenate(true_values, axis=0)

    predicted_values = np.concatenate(predicted_values, axis=0)


    return true_values, predicted_values

true_values, predicted_values = evaluate_model(peft_model, feature_vectors_test)


true_values = true_values.flatten()  
predicted_values = predicted_values.flatten()

df1 = pd.DataFrame()
df1['actual'] = pd.Series(true_values)
df1['predicted'] = pd.Series(predicted_values)

df1.to_csv('/local/data/sdahal_p/covid/result/resulthybrid.csv' , index = False)
