import cv2
import numpy as np
from datetime import datetime
import pandas as pd
import os

def load_id_name_mapping(dataset_dir='dataset'):
    """
    Scan dataset folder and build mapping of ID to Name.
    Expected filename format: Name_ID_index.jpg (e.g. Alice_1_0.jpg)
    """
    mapping = {}
    for file in os.listdir(dataset_dir):
        if file.endswith('.jpg'):
            parts = file.split('_')
            if len(parts) < 3:
                continue
            name = parts[0]
            try:
                id_ = int(parts[1])
                mapping[id_] = name
            except ValueError:
                continue
    return mapping

def log_attendance(name):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    # Only log attendance between 9 AM and 11 AM
    start_time = datetime.strptime("09:00:00", "%H:%M:%S").time()
    end_time = datetime.strptime("11:00:00", "%H:%M:%S").time()
    current_time = now.time()

    if not (start_time <= current_time <= end_time):
        print(f"[WARNING] Attendance for {name} denied outside allowed time: {time_string}")
        return False

    if not os.path.exists('attendance'):
        os.makedirs('attendance')

    filename = f"attendance/attendance_{date_string}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    if not ((df['Name'] == name) & (df['Date'] == date_string)).any():
        new_row = pd.DataFrame({"Name": [name], "Date": [date_string], "Time": [time_string]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)
        print(f"[INFO] Attendance logged for {name} at {time_string} on {date_string}")
        return True
    else:
        print(f"[INFO] Attendance already logged for {name} today.")
        return False

def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    trainer_path = 'trainer/trainer.yml'
    if not os.path.exists(trainer_path):
        print("[ERROR] Trainer model not found. Train the model first.")
        return
    recognizer.read(trainer_path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load ID->Name mapping dynamically from dataset folder
    names = load_id_name_mapping()
    print(f"[INFO] Loaded ID-Name mapping: {names}")

    recognized_today = set()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    cv2.namedWindow("Face Recognition Attendance")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = gray[y:y+h, x:x+w]

            id_, confidence = recognizer.predict(face)
            confidence_percent = 100 - confidence

            if confidence < 60:  # threshold can be adjusted
                name = names.get(id_, "Unknown")
                if name != "Unknown" and name not in recognized_today:
                    if log_attendance(name):
                        recognized_today.add(name)
            else:
                name = "Unknown"

            label_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({round(confidence_percent, 1)}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

        cv2.imshow("Face Recognition Attendance", frame)

        if cv2.waitKey(1) % 256 == 27:  # ESC key to exit
            print("[INFO] Exiting recognition.")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face()
