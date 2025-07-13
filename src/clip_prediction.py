import os
import clip
import torch
from PIL import Image

import cv2 as cv
from facenet_pytorch import MTCNN
from config import *
import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

if CROPPED: 

    mtcnn = MTCNN(
        image_size=None, 
        margin=40,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device=device
    )
    
def rate_clip(img_path: str): 
    img = Image.open(img_path)
    img = preprocess(img)
    img = img.unsqueeze(0).to(device)

    features = {
        "attractiveness": ["unattractive", "attractive"],
        "hotness": ["not hot", "hot"], 
        "aesthetic": ["not aesthetic", "aesthetic"],
        "beauty": ["ugly", "beautiful"],
        "appeal": ["unappealing", "appealing"]
    }
    
    scores = {}
    
    for feature_name, prompts in features.items():
        text = clip.tokenize(prompts).to(device)
        
        with torch.no_grad(): 
            logits_per_image, logits_per_text = model(img, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
        
        scores[feature_name] = probs[1] * 10  # take positive probability * 10
    
    return scores

def get_rating_clip(img_path: str, cropped=CROPPED): 
    if CROPPED: 
        tmp_img = datasets.get_cropped_image(
            path=img_path, 
            model=mtcnn
        )
        
        # temp write image, as it has to be on disk to work here
        if tmp_img is not None:
            tmp_path = img_path.replace(".jpg", "_cropped.jpg")
            cv.imwrite(tmp_path, tmp_img)
            rated_scores = rate_clip(tmp_path)

            
            # remove the cropped image, should not convolute everything
            os.remove(tmp_path)
        else:
            print(f"No face detected in {img_path}, skipping cropping.")
            return None
    
    else:
        rated_scores = rate_clip(img_path)
        
    return rated_scores
        
    
def main(): 
    dir_path = "res/test"
    
    image_files = [f for f in os.listdir(dir_path) 
                   if f.endswith((".jpg", ".png", ".jpeg"))]
    
    # sort numbers based on name, to have photo_i.jpg, where i is in ascending order
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for file_name in image_files:
        img_path = os.path.join(dir_path, file_name)
        print(f"Processing {img_path}")
        scores = get_rating_clip(img_path)
        
        if scores == None: 
            continue
        for feature, score in scores.items():
            print(f"{feature}: {score:.2f}/10")
        
        avg_score = sum(scores.values()) / len(scores)
        print(f"average: {avg_score:.2f}/10")
        print("-" * 40, end="\n\n")
        
main()