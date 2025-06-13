## What is YOLO?

In this project, I have used **YOLOv8 (You Only Look Once)**, which is a powerful object detection model. YOLO detects different objects in an image by looking at the whole image just once. It is very fast and works well in real-time tasks. That's why many people use it in projects like self-driving cars, surveillance, and robotics.

---

## How Object Detection Works

After uploading the image, the YOLO model gives us the following:

- **Bounding Boxes**: The rectangular area showing where the object is.
- **Class ID**: The label of the object, like person, dog, car, etc.
- **Confidence Score**: How sure the model is about its prediction.

These details are used to draw green boxes around detected objects and write the label name with confidence score above each box.

---

## Streamlit Frontend Logic (Simple Explanation)

- The user uploads an image in the app.
- The image is prepared in the correct format and sent to the YOLO model.
- The model runs detection and gives us results like object position, label, and confidence.
- Then, we draw rectangles and labels on the image.
- Finally, the detected image is displayed in the app.

This helps us see which objects the model found and where they are located in the picture.

---

## Webcam and Video Detection

In addition to image detection, the app also supports:

### Webcam Detection
- The user can choose to run real-time object detection using their laptop/computer camera.
- The app reads frames from the webcam (`cv2.VideoCapture(0)`) and continuously runs YOLOv8 detection on each frame.
- Detected frames are displayed live with bounding boxes and labels.

### Video File Detection
- The user can upload a `.mp4`, or `.mov` video file.
- The app processes each frame of the video using the YOLO model.
- It draws boxes and labels, and shows the updated video frame-by-frame in the Streamlit interface.

Both webcam and video modes allow us to apply object detection on moving images, which is more realistic and helpful in many practical use cases.

---
## Setup Instructions

### 1. Clone this repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create virtual environment and activate it
`python3 -m venv venv`
`source venv/bin/activate`

### 3. Install required packages
`pip install -r requirements.txt`

### 4. Install required packages
`streamlit run streamlit_ui.py`


