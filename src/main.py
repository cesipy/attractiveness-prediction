import random

from torch.utils.data import DataLoader, Dataset
import torch
from cnn import CNN

import cnn
import data_processor
from datasets import CustomDataset, process_single_image

from config import *

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
    image_path = "webapp/static/images/im001.jpg"
    result = evaluate_single_photo(
        model=model, 
        img_path="webapp/static/images/im001.jpg"
    )
    print(f"Predicted score for the image {image_path} {result:.3f}")

    image_path = "res/test/im01.jpeg"
    result = evaluate_single_photo(
        model=model, 
        img_path=image_path
    )
    print(f"Predicted score for the image {image_path} {result:.3f}")

    image_path = "res/test/im02.jpeg"
    result = evaluate_single_photo(
        model=model, 
        img_path=image_path
    )
    print(f"Predicted score for the image {image_path} {result:.3f}")

    image_path = "res/test/im03.jpeg"
    result = evaluate_single_photo(
        model=model, 
        img_path=image_path
    )
    print(f"Predicted score for the image {image_path} {result:.3f}")

    image_path = "res/test/im04.jpeg"
    result = evaluate_single_photo(
        model=model, 
        img_path=image_path
    )
    print(f"Predicted score for the image {image_path} {result:.3f}")

if __name__ == "__main__":
    main()

    