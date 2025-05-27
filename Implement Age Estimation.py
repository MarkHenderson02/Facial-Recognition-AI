from deepface import DeepFace
import cv2

def estimate_age(image_path):
    try:
        analysis = DeepFace.analyze(image_path, actions=['age'])
        age = analysis[0]['age']
        return f"Estimated Age: {age}"
    except:
        return "Age Estimation Failed"

# Test
print(estimate_age("sample.jpg"))
