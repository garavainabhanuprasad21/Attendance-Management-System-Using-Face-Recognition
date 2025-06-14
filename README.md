
# Attendance Management System using Face Recognition

## Overview
The **Attendance Management System** is a Python-based application that leverages face recognition technology to automate the process of tracking student attendance. This project uses OpenCV for face detection and recognition, along with a user-friendly graphical interface built with Tkinter. It includes features for capturing images for training, real-time face recognition, and managing attendance records efficiently.

---

## Features
- **Face Recognition**: Automatically recognize students' faces and mark their attendance.
- **Image Capture**: Capture and save images for training the recognition model.
- **Manual Attendance**: Option to manually fill attendance records.
- **CSV Export**: Generate attendance reports in CSV format.
- **Database Integration**: Store attendance records in a MySQL database.

---

## Technologies Used
- **Python**
- **OpenCV**
- **Tkinter**
- **NumPy**
- **Pandas**
- **MySQL**
- **Pillow**

---

## Installation

### 1. Clone the Repository
```bash
https://github.com/garavainabhanuprasad21/Attendance-Management-System-Using-Face-Recognition.git
cd attendance-management-system
```

### 2. Install Required Packages
Ensure Python is installed on your system, then install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set Up MySQL Database
- Create a MySQL database to store attendance records.
- Update the database connection details in the code as needed.

### 4. Download Haarcascade
Download the Haarcascade XML file for face detection from the [OpenCV GitHub repository](https://github.com/opencv/opencv) and place it in the project directory.

---

## Usage

### 1. Capture Images
- Run `gui.py` to open the GUI.
- Enter the student's enrollment number and name.
- Click on **"Take Images"** to capture their face images.

### 2. Train the Model
- After capturing images, click on **"Train Images"** to train the face recognition model.

### 3. Automatic Attendance
- Select **"Recognize Face"** to start the face recognition process using the webcam.

### 4. Manual Attendance
- Use the **"Manually Fill Attendance"** option to manually fill attendance records.

### 5. View Registered Students
- Access the attendance log to view the list of registered students and their details.

---


## Contributing
Contributions are welcome! If you have suggestions for improvements or additional features, feel free to fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
Special thanks to the following:
- **OpenCV** for face detection and recognition functionalities.
- **Tkinter** for GUI development.
- **NumPy** and **Pandas** for data manipulation and analysis.
- **MySQL** for database management.

---

For any questions or issues, feel free to contact **[garavainabhanuprasad21@gmail.com]**.
