import pickle
import os

from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.types import Tensor

import numpy as np
import cv2 as cv
from cv2.typing import MatLike

from config import *

def process_single_image(path: str) -> Tensor: 
    img = cv.imread(path)
    img = cv.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    
    img = np.transpose(img, (2, 0, 1))  # change to CxHxW format (the opencv format)
    img_tensor = torch.from_numpy(img.astype(np.float32))
    
    return img_tensor



class CustomDataset(Dataset): 
    def __init__(self, data: Tuple[str, np.int64], suffix: str): 
        dataset_path = f"{PREFIX_DATASET}_{suffix}.pkl"
        
        print(f"total data_size: {len(data)}")
        
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
        return img, rating
            
        