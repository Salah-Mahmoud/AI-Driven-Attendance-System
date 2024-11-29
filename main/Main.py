import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import cv2

app = Flask(__name__)

# Load the YOLOv8 model (replace with the correct path to your trained weights)
model = YOLO(r"C:\Users\salah\Desktop\FacesClassication\train\best.pt")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_and_classify_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    recognized_faces = []

    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x:x + w]
        face = cv2.resize(cropped_face, (200, 200))
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        results = model.predict(face_gray, save=False, verbose=False)

        for result in results:
            predicted_class_index = result.probs.top1
            predicted_class_name = result.names[predicted_class_index]
            confidence = result.probs.top1conf

            if confidence > 0.7:
                recognized_faces.append({
                    "name": predicted_class_name,
                    "confidence": f"{confidence:.2f}",
                })

    return recognized_faces


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(file.stream).convert('RGB')
    img_np = np.array(image)

    predictions = detect_and_classify_faces(img_np)

    return jsonify({
        'recognized_faces': predictions
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
