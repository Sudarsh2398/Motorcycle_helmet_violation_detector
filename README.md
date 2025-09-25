# Motorcycle_helmet_violation_detector
Motorcycle Helmet Violation Detector

Project Overview

This project, titled "Motorcycle Helmet Violation Detector," is an advanced AI-powered system designed to enhance road safety and traffic enforcement by automatically identifying helmet non-compliance among motorcycle riders. It leverages state-of-the-art computer vision and optical character recognition (OCR) techniques to detect motorcycles, helmets, and license plates in real-time from various sources like images, video streams, and live camera feeds.

Features

•
Real-time Object Detection: Utilizes YOLO (You Only Look Once) for precise and efficient detection of motorcycles, helmets, and license plates.

•
Helmet Non-Compliance Detection: Intelligently determines if a motorcycle rider is not wearing a helmet by analyzing the spatial relationship (Intersection over Union - IoU) between detected helmets and bikes.

•
Enhanced License Plate Recognition (OCR): Employs EasyOCR to accurately extract text from license plates, with specialized validation patterns for Indian license plates.

•
Robust Tracking System: In video mode, the system tracks individual bikes across multiple frames, confirming violations only after a sustained period of helmet non-detection to ensure reliability.

•
Comprehensive Logging: All confirmed violations are meticulously logged to a CSV file, including timestamps and extracted license plate numbers. Visual evidence (images of violations) is also saved.

•
Interactive User Interface: A user-friendly web interface built with Streamlit provides:

•
Image Analysis: Upload images for instant detection and violation reporting.

•
Smart Video Control: Upload video files for frame-by-frame analysis with stop, pause, and resume functionalities.

•
Live Webcam Detection: Real-time monitoring using a live camera feed.



•
Customizable Configuration: Easily adjust parameters such as confidence thresholds, IoU thresholds, and violation confirmation frames.

Technologies Used

•
Streamlit: For creating the interactive web application.

•
YOLO (You Only Look Once): Object detection framework.

•
EasyOCR: Optical Character Recognition library.

•
OpenCV (cv2): For image and video processing.

•
NumPy: For numerical operations.

•
PIL (Pillow): For image manipulation.

•
ffmpeg: For video and audio processing (combining audio with video).

•
Standard Python libraries: os, datetime, csv, tempfile, re, pandas.

Installation

To set up the project locally, follow these steps:

1.
Clone the repository:

2.
Create a virtual environment (recommended):

3.
Install dependencies:

4.
Download the YOLO model weights:
Obtain the bikemodel.pt file (YOLO model weights) and place it in the project root directory. This model is crucial for object detection.

Usage

To run the Streamlit application:

Bash


streamlit run your_main_script_name.py


Replace your_main_script_name.py with the actual name of your Python script containing the Streamlit code (e.g., app.py or main.py).

Once the application is running, open your web browser and navigate to the provided local URL (usually http://localhost:8501). You can then use the different tabs for image analysis, video processing, or live webcam detection.

Configuration

Key configuration parameters can be found and adjusted at the beginning of the main script:

•
MODEL_PATH: Path to your YOLO model (bikemodel.pt).

•
IOU_THRESHOLD: Intersection Over Union threshold.

•
CONF_THRESHOLD: Confidence threshold for bike/plate detection.

•
HELMET_CONF_THRESHOLD: Confidence threshold for helmet detection.

•
HELMET_IOU_THRESHOLD: Minimum IoU for helmet-bike association.

•
VIOLATION_CONFIRM_FRAMES: Frames needed to confirm a violation in video mode.

•
SAVE_VIOLATIONS: Enable/disable saving violation data.

•
OUTPUT_DIR: Directory for saving violation logs and images.

Violation Logging

Violations are logged to violations_log.csv in the OUTPUT_DIR. Each entry includes a timestamp, plate number, source (image/video), and confirmation status. Associated images of violations are also saved in the OUTPUT_DIR.

Contributing

Contributions are welcome! Please feel free to fork the repository, open issues, or submit pull requests.
