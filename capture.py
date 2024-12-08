import cv2
import os

def capture_images(student_id, student_name):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Images")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
            cv2.imwrite(f"dataset/{student_name}.{student_id}.{count}.jpg", face)
            count += 1

        cv2.imshow("Capture Images", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            break
        elif count >= 1:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    student_id = input("Enter student ID: ")
    student_name = input("Enter student name: ")
    capture_images(student_id, student_name)
