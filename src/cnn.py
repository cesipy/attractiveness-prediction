from typing import Optional

import torch
import torchvision

from torchvision.models import  ResNet50_Weights, resnet50, resnet18, ResNet18_Weights, ResNet
from torch import nn
from torch.utils.data import Dataset, DataLoader

from config import *
import data_processor

class CNN(torch.nn.Module): 
    def __init__(self, model_type: str,): 
        super(CNN, self).__init__()
        model: Optional[ResNet]= None
        
        if model_type == "resnet18": 
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_type == "resnet50": 
            model = resnet50(weights=ResNet50_Weights.DEFAULT)    
        else: 
            raise ValueError(f"Model {model_type} not supported. Choose from 'resnet18' or 'resnet50'.")
        
        for param in model.parameters(): 
            param.requires_grad = False

        for param in model.layer4.parameters(): 
            param.requires_grad = True
            
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
    
    def save(self, path: str) -> str:
        """saves model and returns path"""
        torch.save(self.state_dict(), path)
        return path
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    
def train_epoch(model: CNN, data: DataLoader, loss_fn, device, optim):
    total_loss = 0
    num_batches = 0
    
    for batch in data: 
        optim.zero_grad()
        x, y_true = batch
        x = x.to(device)
        y_true = y_true.to(device)
        
        y_pred = model(x)   # shape [batch_size, 1]
        
        y_pred = y_pred.squeeze()  # we only need shape [batch_size], torch.squeeze removes dimensions of size 1
        
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optim.step()
        
        total_loss  += loss.item()
        num_batches += 1
    return total_loss / num_batches


def evaluate(model: CNN, dataloader: DataLoader, loss_fn, device) -> float:
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y_true = batch
            x = x.to(device)
            y_true = y_true.to(device)
            
            y_pred = model(x)       #shape [batch_size, 1]
            
            y_pred = y_pred.squeeze()  # we only need shape [batch_size], torch.squeeze removes dimensions of size 1
                
            loss = loss_fn(y_pred, y_true)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches
    
        
        
def train(model: CNN, train_data: DataLoader, test_data: DataLoader):
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    # loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()

    losses = []
    test_losses   = []

    model = model.to(device)
    
    for i in range(EPOCHS):
        model.train()
        current_loss = train_epoch(
            model=model, 
            data=train_data,
            loss_fn=loss_fn,
            device=device,
            optim=optim
        )
        
        model.eval()
        with torch.no_grad():
            eval_loss = evaluate(
                model=model,
                dataloader=test_data,
                loss_fn=loss_fn,
                device=device
            )
        print(f"Epoch {i+1}/{EPOCHS}, Loss: {current_loss:.4f}, Eval Loss: {eval_loss:.4f}")

        losses.append(current_loss)
        test_losses.append(eval_loss)

    return losses, test_losses