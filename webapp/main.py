from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import csv
from datetime import datetime
from typing import Dict, Any
import json

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cesipy.github.io",  # Your GitHub Pages URL
        "http://localhost:3000",           # Local development only
    ],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_storage = {}

def get_all_images():
    """Get list of all available images"""
    image_dir = "static/images"
    if not os.path.exists(image_dir):
        return []
    
    print(f"Fetching images from {image_dir}"
          )
    res =  [f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(res)
    return res

def save_rating(image_id: str, human_rating: float, session_id: str = "anonymous"):
    """Save rating to CSV file"""
    data = [
        datetime.now().isoformat(),
        session_id, 
        image_id, 
        human_rating
    ]
    
    file_exists = os.path.isfile('ratings.csv')
    with open('ratings.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'session_id', 'image_id', 'human_rating'])
        writer.writerow(data)

@app.get("/")
def root():
    """API status"""
    return {
        "message": "Face Rating API is running!",
        "endpoints": {
            "random_image": "/random-image",
            "submit_rating": "/rate",
            "statistics": "/stats"
        }
    }

@app.get("/random-image")
def get_random_image(session_id: str = "anonymous"):
    """
    Get a random image for rating
    :param session_id: Optional session identifier to track user progress
    :return: Random image data
    """
    all_images = get_all_images()
    
    if not all_images:
        raise HTTPException(status_code=404, detail="No images found in database")
    

    if session_id not in session_storage:
        session_storage[session_id] = {"rated_images": set()}
    
    rated_images = session_storage[session_id]["rated_images"]
    unrated_images = [img for img in all_images if img not in rated_images]
    
    if not unrated_images:
        unrated_images = all_images
        session_storage[session_id]["rated_images"] = set()
    
    # Pick random image
    random_image = random.choice(unrated_images)
    image_url = f"/static/images/{random_image}"
    
    return {
        "success": True,
        "image_url": image_url,
        "image_id": random_image,
        "total_images": len(all_images),
        "remaining_images": len(unrated_images) - 1
    }

@app.post("/rate")
def submit_rating(rating_data: Dict[str, Any]):
    """
    Submit rating for an image
    :param rating_data: {image_id, human_rating, session_id}
    :return: Success confirmation
    """
    image_id = rating_data.get("image_id")
    human_rating = rating_data.get("human_rating")
    session_id = rating_data.get("session_id", "anonymous")
    
    if not image_id:
        raise HTTPException(status_code=400, detail="Missing image_id")
    
    if human_rating is None:
        raise HTTPException(status_code=400, detail="Missing human_rating")
    
    try:
        human_rating = float(human_rating)
        if not (1 <= human_rating <= 10):
            raise ValueError("Rating must be between 1 and 10")
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid rating value")
    

    if session_id not in session_storage:
        session_storage[session_id] = {"rated_images": set()}
    
    session_storage[session_id]["rated_images"].add(image_id)
    
    save_rating(image_id, human_rating, session_id)
    
    return {
        "success": True,
        "message": f"Rating {human_rating} for image {image_id} saved successfully",
        "total_rated": len(session_storage[session_id]["rated_images"])
    }

@app.get("/stats")
def get_statistics():
    """Get rating statistics"""
    if not os.path.exists('ratings.csv'):
        return {
            "total_ratings": 0,
            "unique_images": 0,
            "average_rating": 0,
            "rating_distribution": {}
        }
    
    ratings = []
    with open('ratings.csv', 'r') as file:
        reader = csv.DictReader(file)
        ratings = list(reader)
    
    if not ratings:
        return {
            "total_ratings": 0,
            "unique_images": 0,
            "average_rating": 0,
            "rating_distribution": {}
        }
    
    human_ratings = [float(r['human_rating']) for r in ratings]
    unique_images = len(set(r['image_id'] for r in ratings))
    
    rating_dist = {}
    for rating in human_ratings:
        bucket = int(rating)  # 1-10 buckets
        rating_dist[bucket] = rating_dist.get(bucket, 0) + 1
    
    return {
        "total_ratings": len(ratings),
        "unique_images": unique_images,
        "average_rating": round(sum(human_ratings) / len(human_ratings), 2),
        "rating_distribution": rating_dist,
        "recent_ratings": ratings[-10:]  # Last 10 ratings
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)