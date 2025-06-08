import cv2
import numpy as np
import os

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    data_dir = 'dataset'
    face_samples = []
    ids = []

    if not os.path.exists(data_dir):
        print(f"[ERROR] Dataset folder '{data_dir}' not found. Capture images first.")
        return

    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            path = os.path.join(data_dir, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Could not read image: {path}")
                continue
            try:
                parts = file.split('_')
                if len(parts) < 2:
                    print(f"[WARNING] Unexpected filename format: {file}")
                    continue
                student_id = int(parts[1].split('.')[0])  # handle e.g. "Alice_1.jpg"
            except ValueError:
                print(f"[WARNING] Invalid ID in filename: {file}")
                continue

            # Since image is cropped face already, no need for detectMultiScale here
            face_samples.append(img)
            ids.append(student_id)

    if len(face_samples) == 0:
        print("[ERROR] No face samples found in dataset.")
        return

    print(f"[INFO] Training model with {len(face_samples)} samples.")
    recognizer.train(face_samples, np.array(ids))

    if not os.path.exists('trainer'):
        os.makedirs('trainer')

    recognizer.write('trainer/trainer.yml')
    print(f"[INFO] Model training completed. Trainer saved to 'trainer/trainer.yml'.")

if __name__ == "__main__":
    train_model()
