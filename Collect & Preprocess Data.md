Datasets

Download & Extract Data

mkdir dataset
wget http://example.com/utkface.zip  # Replace with real dataset link
unzip utkface.zip -d dataset/age
wget http://example.com/fer2013.zip
unzip fer2013.zip -d dataset/emotion

Train the Model CNN

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3)

age_model.fit(X_age, y_age, validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stop])

# Save model
age_model.save("age_estimation_model.h5")


Train the Model CNN for Emotion Recognition

emotion_model.fit(X_emotion, y_emotion, validation_split=0.2, epochs=25, batch_size=32, callbacks=[early_stop])

# Save model
emotion_model.save("emotion_recognition_model.h5")

Deploy Models in Real-Time Webcam Application

Load Models

from tensorflow.keras.models import load_model

age_model = load_model("age_estimation_model.h5")
emotion_model = load_model("emotion_recognition_model.h5")

Convert Model for Mobile App Deployment
To use this model on Android:
converter = tf.lite.TFLiteConverter.from_keras_model(age_model)
tflite_model = converter.convert()
with open("age_model.tflite", "wb") as f:
    f.write(tflite_model)

