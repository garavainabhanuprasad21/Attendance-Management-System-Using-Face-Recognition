import cv2
import numpy as np
import os

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    data_dir = 'dataset'
    face_samples = []
    ids = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error: Unable to read image {path}")
                    continue
                student_id = int(file.split('.')[1])  # Extract ID from filename
                faces = face_cascade.detectMultiScale(img)

                for (x, y, w, h) in faces:
                    face_samples.append(img[y:y+h, x:x+w])
                    ids.append(student_id)

    recognizer.train(face_samples, np.array(ids))
    if not os.path.exists('trainer'):
        os.makedirs('trainer')
    recognizer.write('trainer/trainer.yml')
    print(f"[INFO] {len(np.unique(ids))} faces trained.")

if __name__ == "__main__":
    train_model()
