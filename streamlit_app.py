import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image

# Load YOLOv5 model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load a pre-trained YOLOv5 model
    return model

model = load_model()

# Function for object detection
def detect_objects(image):
    results = model(image)
    return results

st.title("🎈 YOLOv5 Object Detection App")
st.write(
    "Upload an image and click 'Detect' to see the objects detected by the YOLOv5 model."
)

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting objects...")

    # Convert the image to a format the YOLO model can process
    image_np = np.array(image)

    # Perform detection
    results = detect_objects(image_np)

    # Display results
    st.write("Detected objects:")
    st.write(results.pandas().xyxy[0])  # Display result in pandas DataFrame format

    # Show annotated image
    annotated_image = results.render()[0]  # Returns an annotated image
    st.image(annotated_image, caption='Annotated Image', use_column_width=True)

if st.button('Detect'):
    if uploaded_file is None:
        st.warning("Please upload an image first!")
