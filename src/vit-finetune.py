import random 
import os
from facenet_pytorch import MTCNN

from typing import Optional

import transformers
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from config import *
import data_processor
from datasets import CustomDataset, process_single_image, process_single_image_face_crop

HIDDEN_UNITS_VIT = 1024

class ViT(nn.Module):
    def __init__(self): 
        super(ViT, self).__init__()
        # register models have 4 additional registers besides patches and cls token
        # absorb more noise, act like "garbage collectors", as i understand it 
        self.backbone = transformers.Dinov2Model.from_pretrained("facebook/dinov2-base") 
        
        for params in self.backbone.parameters():
            params.requires_grad = False
            
        for param in self.backbone.encoder.layer[-3:].parameters():
            param.requires_grad = True
            
        classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=HIDDEN_UNITS_VIT), 
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(in_features=HIDDEN_UNITS_VIT, out_features=HIDDEN_UNITS_VIT//2), 
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(in_features=HIDDEN_UNITS_VIT//2, out_features=1),
        )
        
        self.classifier = classifier
        
    def forward(self, x): 
        features = self.backbone(x).last_hidden_state[:, 0]
        return self.classifier(features)
        
    def save(self, path: str) -> str:
        """saves model and returns path"""
        torch.save(self.state_dict(), path)
        return path
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path))


def train_epoch(model: ViT, data: DataLoader, loss_fn, device, optim):
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


def evaluate(model: ViT, dataloader: DataLoader, loss_fn, device) -> float:
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
        
        
def train(model: ViT, train_data: DataLoader, test_data: DataLoader):
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    # loss_fn = nn.MSELoss()
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

def evaluate_single_photo_cropped(model: ViT, img_path, mtcnn: MTCNN): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_tensor = process_single_image_face_crop(img_path, mtcnn)

    if img_tensor is None:
        return None

    img_tensor = img_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(img_tensor)
        y_pred = y_pred.squeeze().item()
    return y_pred

def evaluate_single_photo(model: ViT, img_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    img_tensor = process_single_image(img_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)   
    model.eval()
        
    with torch.no_grad():
        y_pred = model(img_tensor)
        y_pred = y_pred.squeeze().item() 
    return y_pred

def test_on_photo(model: ViT, path: str, cropping:bool=CROPPED, mtcnn: Optional[MTCNN]=None): 
    if CROPPED:
        result = evaluate_single_photo_cropped(
            model=model, 
            img_path=path, 
            mtcnn=mtcnn
        )
    else:
        result = evaluate_single_photo(
            model=model, 
            img_path=path
        )
    if result:
        result *=2  # to get n/10 instead of n/5
        print(f"Predicted score for the image {path} {result:.3f}/10")
    else: 
        print(f"No face detected in image {path}, skipping...")

def test_on_dir(model: ViT, dir_name): 
    
    image_files = [f for f in os.listdir(dir_name) 
                   if f.endswith((".jpg", ".png", ".jpeg"))]
    
    # sort numbers based on name, to have photo_i.jpg, where i is in ascending order
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # for thispersondoesnotexist
    # image_files.sort(key=lambda x: int(x.split(".")[0]))
    
    for file_name in image_files:
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
            img_path = os.path.join(dir_name, file_name)
            if CROPPED: 
                mtcnn = MTCNN(
                    image_size=None, 
                    margin=40,
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=False,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                test_on_photo(
                    model=model,
                    path=img_path,
                    cropping=CROPPED,
                    mtcnn=mtcnn
                )
            else: 
                test_on_photo(model=model, path=img_path)

def main(): 
    data_scut = data_processor.get_items_scut("res/data_scut", filter=DATASET_FILTER)
    avg_data = data_processor.get_averages(data=data_scut)
    
    data_me_train = data_processor.get_items_mebeauty("res/data_mebeauty/scores/train_crop.csv")
    data_me_test = data_processor.get_items_mebeauty("res/data_mebeauty/scores/test_crop.csv")
    
    data_me = data_me_train + data_me_test
    
    data = avg_data  + data_me
    
    random.shuffle(data)
    train_test_index = int(len(data) * TRAIN_RATIO)
    train_data  = data[:train_test_index]
    test_data = data[train_test_index:]
    
    train_dataset = CustomDataset(train_data, suffix="train", data_augment=USE_DATA_AUGMENTATION)
    test_dataset = CustomDataset(test_data, suffix="test")
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset size: {len(train_dataset)}")
    model = ViT()
    
    train_losses, test_losses = train(
        model=model, 
        train_data=trainloader, 
        test_data=testloader
    )
    data_processor.plot_training_curves(train_losses=train_losses, eval_losses=test_losses)
    saved_model_path = model.save(MODEL_PATH)
    
    # model = ViT()
    # model.load("res/models/model1752614241.pth")

    test_on_dir(model=model, dir_name="res/test")
    
    
if __name__ == "__main__":
    main()