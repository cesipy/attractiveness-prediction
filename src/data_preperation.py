import numpy as np


import dlib

import cv2 as cv

"""
download the model using:

```bash
curl -O https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```
"""

# see documentation
model_path = "res/gan/shape_predictor_68_face_landmarks.dat"
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

TARGET_LANDMARKS = {
    'left_eye': (70, 80),    # Move eyes higher and wider apart
    'right_eye': (186, 80),  # This gives more room for forehead/chin
    'mouth': (128, 160)      # Move mouth up to leave room for chin
}




def get_facial_landmarks( image: np.ndarray):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None

    print(f"len(faces): {len(faces)}")

    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # returns all 68 landmarks, need to extract keypoints(eyes, mouth and right_eye)
    return landmarks

def get_key_points(landmarks: np.ndarray):
    left_eye = np.mean(landmarks[36:42], axis=0)   # Left eye points
    right_eye = np.mean(landmarks[42:48], axis=0)  # Right eye points


    mouth = np.mean(landmarks[48:68], axis=0)      # mouth points

    return {
        'left_eye': left_eye,
        'right_eye': right_eye,
        'mouth': mouth
    }

def normalize_face(image: np.ndarray):

    landmarks = get_facial_landmarks(image)
    if landmarks is None:
        return None

    key_points = get_key_points(landmarks)

    # Source points (detected)
    src = np.array([
        key_points['left_eye'],
        key_points['right_eye'],
        key_points['mouth']
    ], dtype=np.float32)

    # Target points (desired positions)
    dst = np.array([
        TARGET_LANDMARKS['left_eye'],
        TARGET_LANDMARKS['right_eye'],
        TARGET_LANDMARKS['mouth']
    ], dtype=np.float32)

    # Calculate and apply transformation
    transform_matrix = cv.getAffineTransform(src, dst)
    normalized = cv.warpAffine(image, transform_matrix, (256, 256))

    return normalized



if __name__ == "__main__":
    path = "res/data_celeba/050027.jpg"
    img = cv.imread(path)
    landmarks = get_facial_landmarks(img)
    print(landmarks)

    keypoints = get_key_points(landmarks)
    print(keypoints)

    normalized_img = normalize_face(img)
    cv.imwrite("050027_normalized.jpg", normalized_img)

    norm_landmarks = get_facial_landmarks(normalized_img)
    norm_keypoints = get_key_points(norm_landmarks)

    print(norm_keypoints)

