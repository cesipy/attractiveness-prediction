import os
from typing import Optional, Tuple

import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch

from facenet_pytorch import MTCNN

import datasets
from config import *

ratings_name = "All_Ratings.xlsx"

mtcnn = MTCNN(
    image_size=None, 
    margin=40,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def get_items_scut(data_dir: str, filter:str=""): 
    
    ratings =  pd.read_excel(os.path.join(data_dir, ratings_name))
    columns =  ratings.columns

    data:Optional[Tuple[str, int]]= []
    for i in range(len(ratings)):
        item = ratings.iloc[i]
        filename = item.get("Filename")

        if not os.path.exists(os.path.join(SCUT_IMAGE_PATH, filename)):
            continue
        
        if filter in filename:
            full_path = os.path.join(SCUT_IMAGE_PATH, item.get("Filename"))
            data.append((full_path, item.get("Rating")))
    
    return data

def get_items_mebeauty(scores_path:str):
    df = pd.read_csv(scores_path)
    not_found_counter = 0

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
            not_found_counter += 1
            continue
        
        data_list.append((img_path, score))
        
    print(f"total not found images: {not_found_counter}")
    return data_list

def get_items_celeba(fraction=1.):
    data_list = []
    path = "res/data_celeba/cropped"
    
    leng = len(os.listdir(path))
    idx = int(leng * fraction)
    counter = 0
    not_found_counter = 0
    
    for file_name in os.listdir(path):
    
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
            img_path = os.path.join(path, file_name)
            score = SCORE_PLACEHOLDER
            
            
            if not os.path.exists(img_path):
                print(f"Image {img_path} not found, skipping...")
                not_found_counter += 1
                continue
        
        data_list.append((img_path, score))
        counter += 1
        if counter >= idx: 
            break
        
    print(f"total not found images: {not_found_counter}")
    return data_list
    
def get_items_thispersondoesnotexist(path, fraction=1.):
    data_list = []
    idx = int(len(os.listdir(path)) * fraction)
    counter = 0
    
    image_files = [f for f in os.listdir(path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                     and os.path.isfile(os.path.join(path, f))]
    
    for file_name in image_files:
        img_path = os.path.join(path, file_name)
        score = SCORE_PLACEHOLDER
        
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping...")
            continue
        
        data_list.append((img_path, score))
        counter += 1
        if counter >= idx:
            break
        
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
    
    
def crop_scut(): 
    path = "res/data_scut/Images"

    image_files = os.listdir(path)
    print(image_files[:5])
    
    for file_name in image_files:
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
            img_path = os.path.join(path, file_name)
            img_cv = cv.imread(img_path)
            
            if img_cv is None:
                print(f" skipping")
                continue
            
            boxes, _ = mtcnn.detect(img_cv)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = box.astype(int)
                
                margin = 40
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(img_cv.shape[1], x2 + margin)
                y2 = min(img_cv.shape[0], y2 + margin)
                
                face_crop = img_cv[y1:y2, x1:x2]
                
                os.makedirs("res/data_scut/cropped", exist_ok=True)
                cv.imwrite(os.path.join("res/data_scut/cropped", file_name), face_crop)
                print(f"Cropped face saved for {img_path}")
            else:
                print(f"No face detected in {img_path}, skipping...")
    
    
def crop_celeba():
    path = "res/data_celeba"
    output_path = "res/data_celeba/cropped"
    
    image_files = [f for f in os.listdir(path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                   and os.path.isfile(os.path.join(path, f))]
    
    os.makedirs(output_path, exist_ok=True)
    
    for file_name in image_files:
        img_path = os.path.join(path, file_name)
        
        face_crop = datasets.get_cropped_image(img_path, mtcnn)
        
        if face_crop is not None:
            output_file = os.path.join(output_path, file_name)
            cv.imwrite(output_file, face_crop)
            # print(f"Cropped face saved for {file_name}")
        else:
            print(f"No face detected in {file_name}, skipping...")

def crop_mebeauty(): 
    path = "res/data_mebeauty/scores/test_universal_scores.csv"
    
    df = pd.read_csv(path)
    data = []
    
    for idx, row in df.iterrows():
        img_path = "original_" + row[0]
        score = row[1]
        
        # scut is on [0,5]
        score = int(score/2)
        
        parent_path = "res/data_mebeauty"
        img_path = os.path.join(parent_path, img_path)
        
        data.append((img_path, score))
        
    print(f"found {len(data)} images")
    
    
    os.makedirs("res/data_mebeauty/cropped_images", exist_ok=True)
    cropped_data = []
    for i in range(len(data)):
        img_path, score = data[i]
        
        face_crop = datasets.get_cropped_image(img_path, mtcnn)
        filename = img_path.split("/")[-1]
        
        if face_crop is not None:
            output_file = os.path.join("res/data_mebeauty/cropped_images", filename)
            cv.imwrite(output_file, face_crop)
            print(f"Cropped face saved for {filename}")
            
            new_path = f"cropped_images/{filename}"
            cropped_data.append([new_path, score])
        else:
            print(f"No face detected in {filename}, skipping...")

    cropped_df = pd.DataFrame(cropped_data, columns=['image_path', 'score'])
    output_csv = "res/data_mebeauty/scores/test_cropped_scores.csv"
    cropped_df.to_csv(output_csv, index=False)
    
    print(f"Created CSV with {len(cropped_data)} cropped images: {output_csv}")
            

def crop_thispersondoesnotexist(): 
    path = "res/data_thispersondoesnotexist"
    
    image_files = [f for f in os.listdir(path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                     and os.path.isfile(os.path.join(path, f))]
    
    os.makedirs("res/data_thispersondoesnotexist/cropped", exist_ok=True)
    for file_name in image_files:
        img_path = os.path.join(path, file_name)
        
        face_crop = datasets.get_cropped_image(img_path, mtcnn)
        
        if face_crop is not None:
            output_file = os.path.join("res/data_thispersondoesnotexist/cropped", file_name)
            cv.imwrite(output_file, face_crop)
            print(f"Cropped face saved for {file_name}")
        else:
            print(f"No face detected in {file_name}, skipping...")



if __name__ == "__main__":
    # filter = "F"        # only female -F
    #                     # only male   -M
    # data = get_items_scut("res/data_scut", filter=filter)
    
    # avg_data = get_averages(data=data)
    # datasets.CustomDataset(avg_data, suffix="train")
    
    crop_thispersondoesnotexist()
    