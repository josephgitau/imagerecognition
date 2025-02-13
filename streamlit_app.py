import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model.  You can specify a path to a custom trained model here.
model = YOLO('yolov9c.pt')  # Or your preferred model path like 'path/to/best.pt'

# Prediction function. This function performs the object detection.
def predict(chosen_model, img, classes=None, conf=0.5):
    """
    Performs object detection using the provided model and image.

    Args:
        chosen_model: The YOLO model to use.
        img: The image to perform detection on (NumPy array).
        classes: A list of class indices to filter detections (optional).
        conf: The confidence threshold for detections.

    Returns:
        The results object from the model's predict method.
    """
    results = chosen_model.predict(img, classes=classes, conf=conf)
    return results

# Prediction and detection function. This function draws bounding boxes and labels on the image.
def predict_and_detect(chosen_model, img, classes=None, conf=0.5):
    """
    Performs object detection and draws bounding boxes on the image.

    Args:
        chosen_model: The YOLO model to use.
        img: The image to perform detection on (NumPy array).
        classes: A list of class indices to filter detections (optional).
        conf: The confidence threshold for detections.

    Returns:
        A tuple containing the image with bounding boxes and the results object.
    """
    img_copy = img.copy()  # Create a copy of the image to draw on
    results = predict(chosen_model, img_copy, classes, conf=conf)

    for result in results:
        boxes = result.boxes  # Get the Boxes object containing the bounding box information
        for box in boxes:
            xyxy = box.xyxy[0]  # Get the x,y coordinates of the bounding box (top-left and bottom-right)
            cls = int(box.cls)  # Get the class index of the detected object
            conf = float(box.conf[0]) # Get the confidence score of the detection as a float
            label = result.names[cls] # Get the class name from the model's names attribute
            
            x1, y1, x2, y2 = map(int, xyxy) # Convert box coordinates to integers for drawing

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (225, 0, 0), 2)  # Draw the bounding box
            cv2.putText(img_copy, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)  # Add label and confidence

    return img_copy, results


# Streamlit app
st.title("Object Detection with Ultralytics YOLOv8")

# File uploader
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"]) # Specify the image file types accepted

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Class selection (optional)
    class_names = model.names  # Get the class names from the model
    selected_classes = st.multiselect("Select classes to detect (optional)", options=class_names)
    classes_indices = [class_names.index(c) for c in selected_classes] if selected_classes else None # Get indices of selected classes

    # Confidence threshold slider
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # Perform object detection and display the results
    result_img, results = predict_and_detect(model, orig_image, classes=classes_indices, conf=confidence_threshold)

    st.subheader("Original Image")
    st.image(orig_image, caption="Original Image", use_container_width=True) # Use use_container_width

    st.subheader("Detected Objects")
    st.image(result_img, caption="Detected Objects", use_container_width=True)  # Use use_container_width

    # Display results as a Pandas DataFrame (optional)
    if results and results[0].boxes: # Check if there are any results and any detected objects
        data = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0]
            cls = int(box.cls)
            conf = float(box.conf[0])
            label = results[0].names[cls]
            x1, y1, x2, y2 = map(int, xyxy)
            data.append({"Class": label, "Confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        df = pd.DataFrame(data)
        st.dataframe(df) # Display the DataFrame
    elif results and not results[0].boxes: # Handle the case where no objects are detected
        st.write("No objects detected.")
