#no youtube videos, just library methods and such

import torch 
import numpy as np
import matplotlib.pylot as plt
import pandas as pd 


dataset = './data/Midterm_53_group.csv'
frame = pd.read_csv(dataset)

class RegressionNetwork(torch.nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()
        self.linear = torch.nn.Linear(1,1) 
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

network_trafficFinder = RegressionNetwork()




#Data manipulation


#Train with epochs


#Predict results

