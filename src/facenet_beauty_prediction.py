import os
import torch
import random
from typing import Optional
from torch import nn
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

import transformers
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from config import *
import data_processor
from datasets import CustomDataset, process_single_image, process_single_image_face_crop




class FaceRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2').eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            embeddings = self.backbone(x)
        return self.classifier(embeddings)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


def train_epoch(model, dataloader, loss_fn, device, optimizer):
    model.train()
    total_loss = 0

    for x, y_true in dataloader:
        x = x.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()
        y_pred = model(x).squeeze()
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y_true in dataloader:
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model(x).squeeze()
            loss = loss_fn(y_pred, y_true)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train(model, trainloader, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, trainloader, loss_fn, device, optimizer)
        test_loss = evaluate(model, testloader, loss_fn, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Eval Loss: {test_loss:.4f}")

    return model


def evaluate_single_photo(model, img_path, mtcnn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        print(f"No face detected in {img_path}")
        return None

    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = model(face).squeeze().item()
    return y_pred

def test_on_dir(model, dir_name, mtcnn):
    files = [f for f in os.listdir(dir_name) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    files.sort(key=lambda x: int(x.split(".")[0]))  # expects 0.jpg, 1.jpg etc.

    for file in files:
        path = os.path.join(dir_name, file)
        score = evaluate_single_photo(model, path, mtcnn)
        if score is not None:
            print(f"Predicted beauty for {file}: {score*2:.2f}/10")




def main():
    data = data_processor.get_items("res/data_scut", filter=DATASET_FILTER)
    avg_data = data_processor.get_averages(data=data)
    
    random.shuffle(avg_data)
    train_test_index = int(len(avg_data) * TRAIN_RATIO)
    train_data  = avg_data[:train_test_index]
    test_data = avg_data[train_test_index:]
    
    train_dataset = CustomDataset(train_data, suffix="train", data_augment=USE_DATA_AUGMENTATION)
    test_dataset = CustomDataset(test_data, suffix="test")
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = FaceRegressor()
    model = train(model, trainloader, testloader)
    model.save(MODEL_PATH)

    # test on new images
    test_on_dir(model, dir_name="res/data_thispersondoesnotexist", mtcnn=mtcnn)

if __name__ == "__main__":
    main()
