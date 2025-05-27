import cv2
from deepface import DeepFace

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Save frame as image
    cv2.imwrite("temp.jpg", frame)

    # Age & Facial Language Analysis
    age_text = estimate_age("temp.jpg")
    emotion_text = analyze_facial_language("temp.jpg")

    # Display results
    cv2.putText(frame, age_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, emotion_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
