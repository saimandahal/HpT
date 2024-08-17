import pandas as pd

import torch

from torch.utils.data import Dataset, DataLoader

import random

import numpy as np

# Train data

class covid_train(Dataset):

    def __init__(self, csv_file, window_size=50):
        self.data = pd.read_csv(csv_file)

        self.data = self.data[['GISJOIN','confirmed_cases','foot_traffic','Date','deaths','POPDEN','Metro','Micro','POP2','POP3','POP1','POP4','Race1','Race2','MHHI','MNR','MGR','MHV','MHI','QTPOP_percentage','OCCU1']]

        self.data = self.data[self.data['Date'] < 209]


        self.locations = self.data['GISJOIN'].unique()


        self.window_size = window_size

        selected_locations = np.random.choice(self.locations, int(len(self.locations) * 0.02), replace=True)

        self.data = self.data[self.data['GISJOIN'].isin(selected_locations)]

        self.locations = selected_locations
        self.time_series = self.data['Date'].unique()


    def __len__(self):
        
        return len(self.time_series) * (len(self.locations) - self.window_size + 1)


    # Feature generation


  
    def __getitem__(self, idx):
        time_idx = idx // (len(self.locations) - self.window_size + 1)
        loc_idx = idx % (len(self.locations) - self.window_size + 1)
        
        time = self.time_series[time_idx]
        sequence = []
        targets = []
        for loc in self.locations[loc_idx:loc_idx + self.window_size]:
            features = self.data[(self.data['GISJOIN'] == loc) & (self.data['Date'] == time)].iloc[:, 2:].values
            
            target = self.data[(self.data['GISJOIN'] == loc) & (self.data['Date'] == time)].iloc[:, self.data.columns.get_loc('deaths')].values

            # target = self.data[(self.data['GISJOIN'] == loc) & (self.data['Date'] == time)].iloc[:, 2].values
            
            if features.size == 0:
                features = [0] * (len(self.data.columns) - 2) 
                target = 0
            
            sequence.append(features[0])  
            targets.append(target)

        sequence = np.array(sequence)
        targets = np.array(targets)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        return sequence, targets



# Test data 
class covid_test(Dataset):

    def __init__(self, csv_file, window_size=50):

        self.data = pd.read_csv(csv_file)


        self.data = self.data[self.data['Date'] > 209]

        self.data = self.data[['GISJOIN','confirmed_cases','foot_traffic','Date','deaths','POPDEN','Metro','Micro','POP2','POP3','POP1','POP4','Race1','Race2','MHHI','MNR','MGR','MHV','MHI','QTPOP_percentage','OCCU1']]

        self.locations = self.data['GISJOIN'].unique()

        
        self.window_size = window_size


        np.random.seed(0)  
        
        selected_locations = np.random.choice(self.locations, int(len(self.locations) * 0.1), replace=False)
        
        self.data = self.data[self.data['GISJOIN'].isin(selected_locations)]
        
        self.locations = selected_locations
        self.time_series = self.data['Date'].unique()



    def __len__(self):
        return len(self.time_series) * (len(self.locations) - self.window_size + 1)
        

    # Feature generation
    def __getitem__(self, idx):
        
        time_idx = idx // (len(self.locations) - self.window_size + 1)
        
        loc_idx = idx % (len(self.locations) - self.window_size + 1)
        
        time = self.time_series[time_idx]
        sequence = []
        targets = []
        
        for loc in self.locations[loc_idx:loc_idx + self.window_size]:
        
            features = self.data[(self.data['GISJOIN'] == loc) & (self.data['Date'] == time)].iloc[:, 2:].values

            target = self.data[(self.data['GISJOIN'] == loc) & (self.data['Date'] == time)].iloc[:, self.data.columns.get_loc('deaths')].values

            
            if features.size == 0:
        
                features = [0] * (len(self.data.columns) - 2)  
                target = 0
            
            sequence.append(features[0])  
            targets.append(target)

        sequence = np.array(sequence)
        targets = np.array(targets)


        sequence = torch.tensor(sequence, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        return sequence, targets