import streamlit as st
import torch
import numpy as np
from PIL import Image
# Load YOLOv11 model from local file
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov9', 'custom', path='yolo11n.pt', force_reload=True)
    return model

# Initialize the model
model = load_model()

# Function for object detection
def detect_objects(image):
    results = model(image)
    return results

st.title("🎈 YOLOv11 Object Detection App")
st.write("Upload an image and click 'Detect' to see the objects detected by the YOLOv11 model.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
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
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)

if st.button("Detect"):
    if uploaded_file is None:
        st.warning("Please upload an image first!")
