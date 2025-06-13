import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile

st.title("Object detection with YOLO")
st.subheader("Select input source:")
selected_option = st.selectbox(" ", ['Image','Webcam','Video'])

stframe = st.empty()

def load_model():
    model = YOLO('yolov8n.pt')
    return model
model = load_model()

def draw_box(image, results):
    for result in results:
        boxes = result.boxes.xyxy #x1,y1,x2,y2
        scores = result.boxes.conf
        class_ids= result.boxes.cls

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1,y1,x2,y2 = map(int, box)
        label = model.names[int(class_id)]
        confidence = float(score)

        #draw rectangle
        cv2.rectangle(image, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(image, f'{label} {confidence: .2f}', (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
    return image

if selected_option == 'Image':

    st.subheader("Image detection")

    file = st.file_uploader("Upload image: ", type=['png','jpg','jpeg'])

    if file:
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)
    
        st.image(image, caption="Original Image")
        with st.spinner("Running YOLOv8 detection..."):
            results = model.predict(source=image, save=False, conf=0.25)
            result_img = draw_box(image_np.copy(), results)

        st.image(result_img, caption="Detected image")

elif(selected_option == "Webcam"):
    st.subheader("Webcam object detection")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error receiving frame.")
            break

        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # st.image(frame_rgb, caption="captured frame", channels="RGB")

        results = model.predict(source=frame_rgb, save=False, conf=0.25)
        frame = draw_box(frame, results)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        stframe.image(frame)

        if cv2.waitKey(1) == ord('q'):
            break

elif(selected_option == "Video"): 
    st.subheader("Video")
    file = st.file_uploader("Upload video: ", type=["mp4","mov"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error receiving frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow('frame', frame_rgb)

            results = model.predict(source=frame_rgb, save = False, conf=0.25)
            frame = draw_box(frame, results)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame)

            if cv2.waitKey(1) == ord('q'):
                break