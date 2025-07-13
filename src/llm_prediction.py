import os 
from ollama import chat, generate
from facenet_pytorch import MTCNN

import torch
import cv2 as cv

from logger import Logger
import datasets
from config import *


path = "res/test/photo_1.jpg"
model = "qwen2.5vl:7b"
prompt = """Rate the facial attractiveness of the person in this image on a scale of 1-10. 
    
Consider factors like:
- Facial symmetry
- Clear skin
- Proportional features
- Overall aesthetic appeal


Respond in exactly this format:
"X.X/10 (confidence: 0.XX)"
"""

logger = Logger()

if CROPPED: 
    # do it globally, only loaded when this module is used
    # a bit scrappy, but it works
    mtcnn = MTCNN(
        image_size=None, 
        margin=40,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device="cpu"
    )



def get_rating(model, path, logging=True): 
    stream = generate(
        model=model, 
        prompt=prompt,
        #prompt='Rate the attractiveness of the person in this image. Respond with only the following and a confidence score (0-1): "attractive: 0.XX and not attractive: 0.XX", exactly this, nothing else!',
        images=[path], 
    )
    
    rating, description = None, None
    for chunk in stream:
        if chunk[0] == "response":
            #print(chunk[1])
            rating= chunk[1]
            
    stream = generate(
        model=model, 
        prompt='whats in the image? be concise',
        images=[path]
    )

    for chunk in stream:
        if chunk[0] == "response":
            #print(chunk[1])
            description = chunk[1]
            if logging:
                logger.info(f"Processed image: {path}\n\t{rating}\n\tDescription: {description}")
    
def get_rating_llm(img_path, cropped=CROPPED): 
    if CROPPED: 
        tmp_img = datasets.get_cropped_image(
            path=img_path, 
            model=mtcnn
        )
        
        # temp write image, as it has to be on disk to work here
        if tmp_img is not None:
            tmp_path = img_path.replace(".jpg", "_cropped.jpg")
            cv.imwrite(tmp_path, tmp_img)
            rating = get_rating(model=model, path=tmp_path, logging=True)
            # remove the cropped image, should not convolute everything
            os.remove(tmp_path)
        else:
            print(f"No face detected in {img_path}, skipping cropping.")
    
    else:
        rating = get_rating(model=model, path=img_path)
        
    return rating
            
def main(): 
    dir_path = "res/test"
    
    image_files = [f for f in os.listdir(dir_path) 
                   if f.endswith((".jpg", ".png", ".jpeg"))]
    
    # sort numbers based on name, to have photo_i.jpg, where i is in ascending order
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for file_name in image_files:
        img_path = os.path.join(dir_path, file_name)
        print(f"Processing {img_path}")

        rating = get_rating_llm(
            img_path=img_path, 
            cropped=CROPPED
        )
        
        print("rating: ", rating)
        
if __name__ == "__main__":
    main()
            
