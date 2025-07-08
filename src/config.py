from time import time

FC_DIM_SIZE = 1024
DROPOUT_PROB = 0.3
OUTFEATURES = 1
LR = 1e-4
EPOCHS = 10


# where to save the trained model. 
MODEL_PATH = f"res/models/model{int(time())}.pth"

# for the preprocessed pickle when all the images are opened
# see src/datasets.py
PREFIX_DATASET = "res/preprocessed_data"

# path of the scut images
SCUT_IMAGE_PATH = "res/data_scut/Images"