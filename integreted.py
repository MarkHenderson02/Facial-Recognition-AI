import cv2
import numpy as np
import tensorflow as tf
import dlib

# Load trained model
model = tf.keras.models.load_model("personality_model.h5")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def predict_personality(image_path):
    features = extract_facial_features(image_path)
    if features is not None:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0][0]
        
        if prediction > 0.5:
            return "Extrovert"
        else:
            return "Introvert"
    return "No face detected"

# Webcam integration
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_img = frame[y:y+h, x:x+w]

        # Save temporary image and predict
        cv2.imwrite("temp.jpg", face_img)
        personality = predict_personality("temp.jpg")

        # Display result
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, personality, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Psychological Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
