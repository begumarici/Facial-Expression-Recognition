import numpy as np
import cv2
from tensorflow.keras.models import load_model

label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

model = load_model('models/best_model_vgg2.h5')  
cap = cv2.VideoCapture(0)

# Haar Cascade 
yuz_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = yuz_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # take the face
        face_roi = gray[y:y+h, x:x+w]
        resized_frame = cv2.resize(face_roi, (48, 48))
        normalized_frame = resized_frame / 255.0
        input_data = np.reshape(normalized_frame, (1, 48, 48, 1))
        input_data = np.repeat(input_data, 3, axis=-1)  # adapting vgg16

        # make prediction
        prediction = model.predict(input_data)
        predicted_label = np.argmax(prediction)
        label_text = f"Emotion: {label_dict[predicted_label]}"

        # mark the face with red square
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # show the prediction
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # show the result on the screen
    cv2.imshow('Real-Time Prediction', frame)

    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
