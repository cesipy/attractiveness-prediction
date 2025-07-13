import os 
from ollama import chat, generate
from facenet_pytorch import MTCNN

import torch
import cv2 as cv

from logger import Logger
import datasets
from config import *


path = "res/test/photo_1.jpg"
model = "llava"
prompt = """Rate the facial attractiveness of the person in this image on a scale of 1-10. 
    
Consider factors like:
- Facial symmetry
- Clear skin
- Proportional features
- Overall aesthetic appeal


Respond in exactly this format:
"Rating: X.X/10 (confidence: 0.XX)"
"""

logger = Logger()


content_wPath = f"whats in the image, be concise. {path}"
content = "whats in the image, be concise."

# response = chat(
#     model="gemma3:12b", 
#     messages=[{'role': 'user', 'content': content_wPath}]
# )


# response = chat(
#     model="llava:latest", 
#     messages=[{'role': 'user', 'content': content}], 
#     images=[path]
# )


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
            print(chunk[1])
            rating= chunk[1]
            
    stream = generate(
        model=model, 
        prompt='whats in the image? be concise',
        images=[path]
    )

    for chunk in stream:
        if chunk[0] == "response":
            print(chunk[1])
            description = chunk[1]
            if logging:
                logger.info(f"Processed image: {path}\n\t{rating}\n\tDescription: {description}")
    
            
def main(): 
    dir_path = "res/test"
    
    if CROPPED: 
        mtcnn = MTCNN(
                    image_size=None, 
                    margin=40,
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=False,
                    device="cpu"
                )
    
    # Get all image files and sort them naturally
    image_files = [f for f in os.listdir(dir_path) 
                   if f.endswith((".jpg", ".png", ".jpeg"))]
    
    # Sort naturally (handles numbers correctly)
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for file_name in image_files:
        img_path = os.path.join(dir_path, file_name)
        print(f"Processing {img_path}")
        
        if CROPPED: 
            tmp_img = datasets.get_cropped_image(
                path=img_path, 
                model=mtcnn
            )
            
            # temp write image 
            if tmp_img is not None:
                tmp_path = img_path.replace(".jpg", "_cropped.jpg")
                cv.imwrite(tmp_path, tmp_img)
                get_rating(model=model, path=tmp_path, logging=True)
                # remove
                os.remove(tmp_path)
            else:
                print(f"No face detected in {img_path}, skipping cropping.")
        
        else:
            get_rating(model=model, path=img_path)
            
        
        print("-" * 20)
            
        
            
main()
            
