import clip
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random

import datasets, data_processor
from datasets import CustomDataset
from config import *



class CLIP(nn.Module): 
    def __init__(self, ): 
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = clip.load(CLIP_MODEL, device=device)[0]
        self.preprocess = clip.load(CLIP_MODEL, device=device)[1]
        
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # for param in self.clip.visual.transformer.resblocks[-1].parameters():
        #     param.requires_grad = True
            
        classifier = [
            nn.Linear(in_features=768, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.1), 
            nn.Linear(in_features=128, out_features=1),
            # nn.Sigmoid()      #TODO: add sigmoid and scale it up in the code
        ]
        
        self.fc = nn.Sequential(
            *classifier
        )
        
    def forward(self, x):
        x = self.clip.encode_image(x)     # this is done in the preprocessing step now
        x = x.float()
        x = self.fc(x)
        return x.squeeze()

def train_epoch(model: CLIP, data: DataLoader, loss_fn, device, optim):
    total_loss = 0
    num_batches = 0
    
    for batch in data: 
        optim.zero_grad()
        x, y_true = batch
        x = x.to(device)
        y_true = y_true.to(device)
        
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def evaluate(model: CLIP, dataloader: DataLoader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y_true = batch
            x = x.to(device)
            y_true = y_true.to(device)
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train(model: CLIP, train_data: DataLoader, test_data: DataLoader):
    
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.clip = model.clip.to(device)
    loss_fn = nn.SmoothL1Loss()

    losses = []
    test_losses = []
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
    
def main(): 
    data = data_processor.get_items_scut("res/data_scut", filter=DATASET_FILTER)
    avg_data = data_processor.get_averages(data=data)
    
    
    random.shuffle(avg_data)
    train_test_index = int(len(avg_data) * TRAIN_RATIO)
    train_data  = avg_data[:train_test_index]
    test_data = avg_data[train_test_index:]
    
    train_dataset = CustomDataset(train_data, suffix="train_clip", data_augment=True, clip_preprocess=True)
    test_dataset  = CustomDataset(test_data, suffix="test_clip", clip_preprocess=True)
    trainloader   = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader    = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CLIP()
    train_losses, test_losses = train(
        model=model, 
        train_data=trainloader, 
        test_data=testloader
    )
    
    
if __name__ == "__main__":
    main()