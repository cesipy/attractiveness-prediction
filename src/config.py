from time import time

FC_DIM_SIZE = 1024
DROPOUT_PROB = 0.5
OUTFEATURES = 1
LR = 4e-5
EPOCHS = 100
BATCH_SIZE = 32


# where to save the trained model. 
MODEL_PATH = f"res/models/model{int(time())}.pth"

# for the preprocessed pickle when all the images are opened
# see src/datasets.py
PREFIX_DATASET = "res/preprocessed_data/preprocessed_data"

# path of the scut images
SCUT_IMAGE_PATH = "res/data_scut/Images"

# what amount of data set to use for training
TRAIN_RATIO = 0.8

IMAGE_SIZE = (224,224)

MODEL_TYPE = "resnet18"


# for the dataset
# female - "F"
# male   - "M"
# none   - ""
DATASET_FILTER = ""       
USE_DATA_AUGMENTATION = True 

CROPPED = True    # crop the image using facent/mntt

CLIP_MODEL = "ViT-L/14@336px"