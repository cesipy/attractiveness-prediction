import os
from typing import Optional, Tuple

import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import datasets
from config import *

ratings_name = "All_Ratings.xlsx"

def get_items_scut(data_dir: str, filter:str=""): 
    
    ratings =  pd.read_excel(os.path.join(data_dir, ratings_name))
    columns =  ratings.columns

    data:Optional[Tuple[str, int]]= []
    for i in range(len(ratings)):
        item = ratings.iloc[i]
        filename = item.get("Filename")
        
        if filter in filename:
            full_path = os.path.join(SCUT_IMAGE_PATH, item.get("Filename"))
            data.append((full_path, item.get("Rating")))
    
    return data

def get_items_mebeauty(scores_path:str):
    df = pd.read_csv(scores_path)

    data_list = []
    
    for idx, row in df.iterrows():
        img_path = row[0]
        score = row[1]
        
        # scut is on [0,5]
        score = int(score/2)
        
        parent_path = "res/data_mebeauty"
        img_path = os.path.join(parent_path, img_path)
        # print(f"Image path: {img_path}, score: {score}")
        
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping...")
            continue
        
        data_list.append((img_path, score))
        
    return data_list

def get_averages(data: Tuple[str, int]) -> Tuple[float, float]:
    hash_map = {}
    avg_data = []
    
    for dp in data: 
        hash_map[dp[0]] = []
        
    for dp in data: 
        hash_map[dp[0]].append(dp[1])
        
    for key in hash_map: 
        hash_map[key] = np.round(
            sum(hash_map[key]) / len(hash_map[key]), 
            decimals=3
        )
        avg_data.append((key, hash_map[key]))
        
        
    return avg_data
    
    
def plot_training_curves(train_losses, eval_losses):
    """Plot training and evaluation losses"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, eval_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss difference (overfitting indicator)
    plt.subplot(1, 2, 2)
    loss_diff = [eval_loss - train_loss for train_loss, eval_loss in zip(train_losses, eval_losses)]
    plt.plot(epochs, loss_diff, 'g-', label='Val Loss - Train Loss', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Indicator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("res/training_curves.png", dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    filter = "F"        # only female -F
                        # only male   -M
    data = get_items_scut("res/data_scut", filter=filter)
    
    avg_data = get_averages(data=data)
    datasets.CustomDataset(avg_data, suffix="train")