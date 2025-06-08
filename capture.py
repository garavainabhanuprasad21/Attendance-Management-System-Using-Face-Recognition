import cv2
import os

def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    name = input("Enter your name (e.g., Alice): ").strip()
    id_input = input("Enter your numeric ID (e.g., 1): ").strip()
    try:
        student_id = int(id_input)
    except ValueError:
        print("[ERROR] Invalid ID. Must be a number.")
        cam.release()
        return

    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    print("Press SPACE to capture your face image, ESC to exit.")

    captured = False
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("Exiting capture.")
            break
        elif key == 32 and not captured:  # SPACE key, capture only once
            if len(faces) == 0:
                print("[WARNING] No face detected. Try again.")
                continue

            # Save the first detected face region in grayscale
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]

            filename = f"{name}_{student_id}.jpg"
            filepath = os.path.join(dataset_dir, filename)

            # Save the image
            cv2.imwrite(filepath, face_img)
            print(f"[INFO] Image saved as {filepath}")
            captured = True

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
