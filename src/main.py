import random
import os
from typing import Optional

from torch.utils.data import DataLoader, Dataset
import torch
from cnn import CNN
from facenet_pytorch import MTCNN
from PIL import Image


import cnn
import data_processor
from datasets import CustomDataset, process_single_image, process_single_image_face_crop

from config import *



def evaluate_single_photo_cropped(model: CNN, img_path, mtcnn: MTCNN): 
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

def evaluate_single_photo(model: CNN, img_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    img_tensor = process_single_image(img_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)   
    model.eval()
        
    with torch.no_grad():
        y_pred = model(img_tensor)
        y_pred = y_pred.squeeze().item() 
    return y_pred

def test_on_photo(model: CNN, path: str, cropping:bool=CROPPED, mtcnn: Optional[MTCNN]=None): 
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

def test_on_dir(model: CNN, dir_name): 
    # counter = 0
    # for file_name in os.listdir(dir_name):
    #     if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
    #         counter += 1
    #         # rename the file 
    #         new_name = f"photo_{counter}.jpg"
    #         old_path = os.path.join(dir_name, file_name)
    #         new_path = os.path.join(dir_name, new_name)
    #         os.rename(old_path, new_path)

    
    for file_name in os.listdir(dir_name):
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
    
    print(f"Dataset size: {len(train_dataset)}")
    model = CNN(model_type=MODEL_TYPE)
    
    train_losses, test_losses = cnn.train(
        model=model, 
        train_data=trainloader, 
        test_data=testloader
    )
    data_processor.plot_training_curves(train_losses=train_losses, eval_losses=test_losses)

    saved_model_path = model.save(MODEL_PATH)

    model = CNN(model_type=MODEL_TYPE)
    model.load(saved_model_path)


    test_on_dir(model=model, dir_name="res/test/")

if __name__ == "__main__":
    main()

    