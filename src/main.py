import random

from torch.utils.data import DataLoader, Dataset

from cnn import CNN

import cnn
import data_processor
from datasets import CustomDataset

from config import *

def main(): 
    data = data_processor.get_items("res/data_scut")
    avg_data = data_processor.get_averages(data=data)
    
    random.shuffle(avg_data)
    train_test_index = int(len(avg_data) * TRAIN_RATIO)
    train_data  = avg_data[:train_test_index]
    test_data = avg_data[train_test_index:]
    
    train_dataset = CustomDataset(train_data, suffix="train")
    test_dataset = CustomDataset(test_data, suffix="test")
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset size: {len(train_dataset)}")
    model = CNN(model_type=MODEL_TYPE)
    
    cnn.train(
        model=model, 
        train_data=trainloader, 
        test_data=testloader
    )


if __name__ == "__main__":
    main()

    