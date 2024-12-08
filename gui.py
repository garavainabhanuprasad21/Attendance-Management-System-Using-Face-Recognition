import tkinter as tk
from tkinter import messagebox
import subprocess
import pandas as pd
import os
from datetime import datetime

def run_capture_images():
    subprocess.run(["python", "capture.py"])

def run_train_model():
    subprocess.run(["python", "train.py"])

def run_recognize_face():
    subprocess.run(["python", "recognize.py"])

def manual_attendance():
    def submit_attendance():
        student_id = entry_id.get()
        student_name = entry_name.get()

        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")

        if not os.path.exists('attendance'):
            os.makedirs('attendance')

        filename = f"attendance/attendance_{date_string}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])

        new_row = pd.DataFrame({"Name": [student_name], "Date": [date_string], "Time": [time_string]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)
        messagebox.showinfo("Success", f"Manually logged attendance for {student_name}.")

    manual_window = tk.Toplevel(app)
    manual_window.title("Manual Attendance")

    label_id = tk.Label(manual_window, text="Student ID:")
    label_id.pack(pady=5)
    entry_id = tk.Entry(manual_window)
    entry_id.pack(pady=5)

    label_name = tk.Label(manual_window, text="Student Name:")
    label_name.pack(pady=5)
    entry_name = tk.Entry(manual_window)
    entry_name.pack(pady=5)

    submit_button = tk.Button(manual_window, text="Submit", command=submit_attendance)
    submit_button.pack(pady=10)

app = tk.Tk()
app.title("Attendance Management System")

capture_button = tk.Button(app, text="Capture Images", command=run_capture_images)
capture_button.pack(pady=10)

train_button = tk.Button(app, text="Train Model", command=run_train_model)
train_button.pack(pady=10)

recognize_button = tk.Button(app, text="Recognize Face", command=run_recognize_face)
recognize_button.pack(pady=10)

manual_button = tk.Button(app, text="Manual Attendance", command=manual_attendance)
manual_button.pack(pady=10)

app.mainloop()
