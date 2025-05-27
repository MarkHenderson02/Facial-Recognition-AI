def analyze_facial_language(image_path):
    try:
        analysis = DeepFace.analyze(image_path, actions=['emotion'])
        emotion = analysis[0]['dominant_emotion']
        return f"Facial Language: {emotion}"
    except:
        return "Facial Language Analysis Failed"

# Test
print(analyze_facial_language("sample.jpg"))
