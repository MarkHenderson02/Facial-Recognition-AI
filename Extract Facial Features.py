import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_facial_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        features = []
        
        # Example: Extract jawline and nose width
        for i in range(0, 68):  
            x, y = landmarks.part(i).x, landmarks.part(i).y
            features.append((x, y))

        return np.array(features).flatten()  # Convert to 1D array

    return None  # No face detected

# Example usage
features = extract_facial_features("sample.jpg")
print(features)
