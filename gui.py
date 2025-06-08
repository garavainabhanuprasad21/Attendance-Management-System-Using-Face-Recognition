import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import pandas as pd
import os
from datetime import datetime
import cv2


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Attendance Management System")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')

        # Tabs
        self.tab_capture = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_recognize = ttk.Frame(self.notebook)
        self.tab_manual = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_capture, text='Capture Images')
        self.notebook.add(self.tab_train, text='Train Model')
        self.notebook.add(self.tab_recognize, text='Recognize Face')
        self.notebook.add(self.tab_manual, text='Manual Attendance')

        # Build each tab
        self.build_capture_tab()
        self.build_train_tab()
        self.build_recognize_tab()
        self.build_manual_tab()

        # Status log
        self.status_log = scrolledtext.ScrolledText(root, height=8, state='disabled', font=("Consolas", 10))
        self.status_log.pack(fill='x', padx=10, pady=5)

    def log(self, message):
        self.status_log['state'] = 'normal'
        self.status_log.insert(tk.END, message + "\n")
        self.status_log.see(tk.END)
        self.status_log['state'] = 'disabled'

    # --------- Capture Tab ----------
    def build_capture_tab(self):
        frm = self.tab_capture

        ttk.Label(frm, text="Enter Student ID:", font=("Arial", 12)).pack(pady=5)
        self.entry_id = ttk.Entry(frm, font=("Arial", 12))
        self.entry_id.pack(pady=5)

        ttk.Label(frm, text="Enter Student Name:", font=("Arial", 12)).pack(pady=5)
        self.entry_name = ttk.Entry(frm, font=("Arial", 12))
        self.entry_name.pack(pady=5)

        self.btn_start_capture = ttk.Button(frm, text="Start Capture", command=self.start_capture)
        self.btn_start_capture.pack(pady=10)

    def start_capture(self):
        student_id = self.entry_id.get().strip()
        student_name = self.entry_name.get().strip()

        if not student_id or not student_name:
            messagebox.showerror("Input Error", "Please enter both Student ID and Name.")
            return

        self.log(f"[INFO] Starting capture for {student_name} (ID: {student_id})...")
        self.capture_images(student_id, student_name)

    def capture_images(self, student_id, student_name):
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            self.log("[ERROR] Could not open webcam.")
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.log("[INFO] Press SPACE to capture image, ESC to cancel.")

        captured = False

        while True:
            ret, frame = cam.read()
            if not ret:
                self.log("[ERROR] Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(f"Capture Images for {student_name}", frame)
            key = cv2.waitKey(1)

            if key % 256 == 27:  # ESC
                self.log("[INFO] Capture cancelled.")
                break
            elif key % 256 == 32:  # SPACE pressed
                if len(faces) == 0:
                    self.log("[WARNING] No face detected. Try again.")
                    continue

                # Save image with unique name
                x, y, w, h = faces[0]  # Take only first detected face
                face = gray[y:y + h, x:x + w]

                if not os.path.exists('dataset'):
                    os.makedirs('dataset')

                base_name = f"{student_name}_{student_id}"
                file_index = 0
                while True:
                    filename = f"dataset/{base_name}_{file_index}.jpg"
                    if not os.path.exists(filename):
                        break
                    file_index += 1

                cv2.imwrite(filename, face)
                self.log(f"[INFO] Image saved: {filename}")
                captured = True
                break  # Exit loop after capturing one image

        cam.release()
        cv2.destroyAllWindows()

        if captured:
            messagebox.showinfo("Success", f"Image captured and saved for {student_name}.")
        else:
            messagebox.showinfo("Info", "No image captured.")

    # --------- Train Tab ----------
    def build_train_tab(self):
        frm = self.tab_train
        ttk.Label(frm, text="Train the model with captured images", font=("Arial", 14)).pack(pady=10)

        self.btn_train = ttk.Button(frm, text="Train Model", command=self.run_train_model)
        self.btn_train.pack(pady=10)

    def run_train_model(self):
        self.log("[INFO] Training started...")
        try:
            subprocess.run(["python", "train.py"], check=True)
            self.log("[INFO] Training completed successfully.")
            messagebox.showinfo("Training", "Model training completed successfully.")
        except subprocess.CalledProcessError as e:
            self.log(f"[ERROR] Training failed: {e}")
            messagebox.showerror("Training Error", "Training failed. See logs.")

    # --------- Recognize Tab ----------
    def build_recognize_tab(self):
        frm = self.tab_recognize
        ttk.Label(frm, text="Start face recognition and attendance logging", font=("Arial", 14)).pack(pady=10)

        self.btn_recognize = ttk.Button(frm, text="Start Recognition", command=self.run_recognize_face)
        self.btn_recognize.pack(pady=10)

    def run_recognize_face(self):
        self.log("[INFO] Starting face recognition...")
        try:
            subprocess.run(["python", "recognize.py"], check=True)
            self.log("[INFO] Face recognition ended.")
        except subprocess.CalledProcessError as e:
            self.log(f"[ERROR] Recognition failed: {e}")
            messagebox.showerror("Recognition Error", "Recognition failed. See logs.")

    # --------- Manual Attendance Tab ----------
    def build_manual_tab(self):
        frm = self.tab_manual

        ttk.Label(frm, text="Enter Student ID:", font=("Arial", 12)).pack(pady=5)
        self.manual_entry_id = ttk.Entry(frm, font=("Arial", 12))
        self.manual_entry_id.pack(pady=5)

        ttk.Label(frm, text="Enter Student Name:", font=("Arial", 12)).pack(pady=5)
        self.manual_entry_name = ttk.Entry(frm, font=("Arial", 12))
        self.manual_entry_name.pack(pady=5)

        self.btn_manual_submit = ttk.Button(frm, text="Submit Attendance", command=self.manual_attendance)
        self.btn_manual_submit.pack(pady=10)

    def manual_attendance(self):
        student_id = self.manual_entry_id.get().strip()
        student_name = self.manual_entry_name.get().strip()

        if not student_id or not student_name:
            messagebox.showerror("Input Error", "Please enter both Student ID and Name.")
            return

        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")

        # Time restriction: Only between 9 AM and 11 AM
        start_time = datetime.strptime("09:00:00", "%H:%M:%S").time()
        end_time = datetime.strptime("11:00:00", "%H:%M:%S").time()
        current_time = now.time()

        if not (start_time <= current_time <= end_time):
            messagebox.showwarning("Attendance Time Restriction", "Attendance can only be logged between 9 AM and 11 AM.")
            return

        if not os.path.exists('attendance'):
            os.makedirs('attendance')

        filename = f"attendance/attendance_{date_string}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])

        # Check if attendance already logged for this student today
        if ((df['Name'] == student_name) & (df['Date'] == date_string)).any():
            messagebox.showinfo("Duplicate Attendance", f"Attendance already logged for {student_name} today.")
            return

        new_row = pd.DataFrame({"Name": [student_name], "Date": [date_string], "Time": [time_string]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)

        self.log(f"[INFO] Manually logged attendance for {student_name} at {time_string} on {date_string}")
        messagebox.showinfo("Success", f"Attendance logged for {student_name}.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
