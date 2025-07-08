from typing import Optional

import torch
import torchvision

from torchvision.models import  ResNet50_Weights, resnet50, resnet18, ResNet18_Weights, ResNet
from torch import nn
from torch.utils.data import Dataset, DataLoader

from config import *


class CNN(torch.nn.Module): 
    def __init__(self, model: str,): 
        super(CNN, self).__init__()
        model: Optional[ResNet]= None
        if model == "resnet18": 
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model == "resnet50": 
            model = resnet50(weights=ResNet50_Weights.DEFAULT)    
        else: 
            raise ValueError(f"Model {model} not supported. Choose from 'resnet18' or 'resnet50'.")
        
        for param in model.parameters(): 
            param.requires_grad = False
            
        fc = [
            nn.Linear(in_features=model.fc.in_features, out_features=FC_DIM_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_PROB), 
            nn.Linear(in_features=FC_DIM_SIZE, out_features=FC_DIM_SIZE//2), 
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(in_features=FC_DIM_SIZE//2, out_features=OUTFEATURES),
        ]
        
        model.fc = nn.Sequential(*fc)
        
        self.model = model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    
def train_epoch(model: CNN, data: DataLoader):
    
    for batch in data: 
        x, y_true = batch
        
        y_pred = model(x)
        # TODO: how to incorporate it here?
        
        
        
        
        
        
def train(model: CNN, train_data: DataLoader, test_data: DataLoader):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i in range(EPOCHS)
    
    