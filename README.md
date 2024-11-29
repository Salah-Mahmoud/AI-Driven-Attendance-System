# AI-Driven Attendance System

This project implements an AI-powered attendance system that uses face detection and classification to register attendance. The system utilizes **Haar Cascade** for face detection and **YOLOv11** for face classification, along with **Flask** to build an API for attendance management.

## Features

- **Face Detection:** Uses Haar Cascade to detect and crop faces from input images.
- **Face Classification:** YOLOv11 classifies the detected faces.
- **Attendance System:** Built with Flask, it tracks and records attendance based on detected and classified faces.

### Requirements

- Python
- OpenCV
- Flask
- YOLOv11 pre-trained weights.
- Haar Cascade `haarcascade_frontalface_default.xml` (for face detection)
