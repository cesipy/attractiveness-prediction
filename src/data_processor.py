import os
from typing import Optional, Tuple

import pandas as pd
import cv2 as cv

import datasets
from config import *

ratings_name = "All_Ratings.xlsx"

def get_items(data_dir: str): 
    
    ratings =  pd.read_excel(os.path.join(data_dir, ratings_name))
    columns =  ratings.columns
    
    print(columns)

    data:Optional[Tuple[str, int]]= []
    for i in range(len(ratings)):
        item = ratings.iloc[i]
        full_path = os.path.join(SCUT_IMAGE_PATH, item.get("Filename"))
        data.append((full_path, item.get("Rating")))
    
    return data
    
    
if __name__ == "__main__":
    data = get_items("res/data_scut")
    print(len(data))
    # datasets.CustomDataset(data, suffix="train")