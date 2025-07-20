from time import time

FC_DIM_SIZE = 1024
DROPOUT_PROB = 0.3
OUTFEATURES = 1
LR = 1.5e-5
EPOCHS = 25
BATCH_SIZE = 256


# where to save the trained model. 
MODEL_PATH = f"res/models/model{int(time())}.pth"

# for the preprocessed pickle when all the images are opened
# see src/datasets.py
PREFIX_DATASET = "res/preprocessed_data/preprocessed_data"

# what amount of data set to use for training
TRAIN_RATIO = 0.8

IMAGE_SIZE = (224,224)

MODEL_TYPE = "resnet18"


# for the dataset SCUT
# female - "F"
# male   - "M"
# none   - ""
DATASET_FILTER = ""       
USE_DATA_AUGMENTATION = True 

# path of the scut images
SCUT_IMAGE_PATH = "res/data_scut/Images"

CROPPED = False    # crop the image using facent/mntt

CLIP_MODEL = "ViT-L/14@336px"

# for the dataset mebeauty
ME_PARENT_PATH = "res/data_mebeauty"
