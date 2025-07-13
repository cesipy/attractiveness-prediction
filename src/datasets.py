import pickle
import os

from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from torch import Tensor
import matplotlib.pyplot as plt

import torchvision.transforms.functional as tf

import numpy as np
import cv2 as cv
from cv2.typing import MatLike

from PIL import Image
from config import *

# mtcnn = 

def process_single_image(path: str) -> Tensor: 
    img = cv.imread(path)
    img = cv.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    
    img = np.transpose(img, (2, 0, 1))  # change to CxHxW format (the opencv format)
    img_tensor = torch.from_numpy(img.astype(np.float32))
    
    return img_tensor

def process_single_image_face_crop(path: str, model) -> Tensor:
    """
    Use MTCNN for detection but match training preprocessing exactly
    """
    # Load image the same way as training (OpenCV = BGR)
    img_cv = cv.imread(path)
    img_pil = Image.open(path).convert('RGB')
    
    # Use MTCNN just for face detection (get bounding boxes)
    boxes, probs = model.detect(img_pil)
    
    if boxes is None:
        return None
    
    # Get the best face (highest confidence)
    best_box_idx = np.argmax(probs)
    box = boxes[best_box_idx]
    
    # Add margin (similar to MTCNN's margin)
    margin = 40
    x1, y1, x2, y2 = box
    
    # Add margin
    x1 = max(0, int(x1 - margin))
    y1 = max(0, int(y1 - margin))
    x2 = min(img_cv.shape[1], int(x2 + margin))
    y2 = min(img_cv.shape[0], int(y2 + margin))
    
    # Crop face using OpenCV (same as training)
    face_crop = img_cv[y1:y2, x1:x2]
    
    # Apply EXACT same preprocessing as training
    face_crop = cv.resize(face_crop, IMAGE_SIZE)  # Use your IMAGE_SIZE
    face_crop = face_crop.astype(np.float32) / 255.0
    face_crop = np.transpose(face_crop, (2, 0, 1))
    face_tensor = torch.from_numpy(face_crop.astype(np.float32))
    
    # Save for visualization (convert BGR to RGB for saving)
    face_crop_rgb = cv.cvtColor((face_crop.transpose(1, 2, 0) * 255).astype(np.uint8), cv.COLOR_BGR2RGB)
    Image.fromarray(face_crop_rgb).save(f"{path}_cropped_face.jpg")
    
    return face_tensor


class CustomDataset(Dataset): 
    def __init__(
        self, 
        data: Tuple[str, np.int64], 
        suffix: str, 
        data_augment: bool= False
        ): 
        dataset_path = f"{PREFIX_DATASET}_{suffix}.pkl"
        
        print(f"total data_size: {len(data)}")

        self.transform = None

        if data_augment: 
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomVerticalFlip(p=0.3),           
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                
                # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                # transforms.RandomZoomOut(scale=(0.8, 1.2)),
                # transforms.RandomRotation(degrees=45),  # Very aggressive - try only if needed
            ])
        
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                self.data = pickle.load(f)
                
            print("found preprocessed data, finished loading it!")
        else: 
            print("preprocessed data not found, preprocessing...")
            data = self.__process_dataset(data=data)
            
            with open(dataset_path, 'wb') as f:
                pickle.dump(data, f)
            self.data = data
            
            
    def __process_dataset(self, data: Tuple[str, np.int64]): 
        """preprocesses the dataset, and saves it as pickle"""
        
        processed_data = []
        counter = 0
        
        for dp in data: 
            counter += 1
            img_path, img_rating = dp
            processed_img = process_single_image(img_path)
            rating_tensor = torch.tensor(img_rating, dtype=torch.float32)
            processed_data.append((processed_img, rating_tensor))
            
            if counter % 300 == 0:
                print(f"processed {counter} images")
                
        return processed_data

            
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        img, rating = self.data[idx]

        if self.transform: 
            img = self.transform(img)

        return img, rating
            
        