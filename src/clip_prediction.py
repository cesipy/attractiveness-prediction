import os
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

def get_rating_clip(img_path: str): 
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
            
        print(f"{feature_name}: {probs}")
        scores[feature_name] = probs[1] * 10  # take positive probability * 10
    
    return scores
    
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
        
        for feature, score in scores.items():
            print(f"{feature}: {score:.2f}/10")
        
        avg_score = sum(scores.values()) / len(scores)
        print(f"average: {avg_score:.2f}/10")
        print("-" * 40, end="\n\n")
        
main()