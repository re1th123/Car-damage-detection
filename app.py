import streamlit as st
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
import cv2
import numpy as np
from PIL import Image
from detectron2 import model_zoo
import tracemalloc
import os

# Enable environment variable to prevent duplicate OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optional: Enable memory tracing
tracemalloc.start()

# Load model and dataset metadata
@st.cache_resource
def load_predictor(model_choice, confidence_threshold):
    dataset_dir = "C:/Users/revan/Downloads/cardamge"
    img_dir = "img"
    val_dir = "val"
    register_coco_instances("car_dataset_val", {}, os.path.join(dataset_dir, val_dir, "COCO_val_annos.json"), os.path.join(dataset_dir, img_dir))

    cfg = get_cfg()

    if model_choice == "Faster R-CNN":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "model_fina.pth"
    elif model_choice == "Mask R-CNN":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "model_fi.pth"
    elif model_choice == "RetinaNet":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "mdferewf.h5"

    cfg.DATASETS.TEST = ("car_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold  # Set confidence threshold here
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if using a GPU

    predictor = DefaultPredictor(cfg)
    return predictor

# Function to classify damage severity based on area
def classify_damage_severity(area):
    if area < 1000:
        return "Minor Damage"
    elif area < 5000:
        return "Moderate Damage"
    else:
        return "Severe Damage"

st.title("Car Damage Detection ")
st.write("Upload an image, select a model, set the confidence threshold, and view AI-based car damage detection and classification results.")

# Expandable AI Section: Information on model choices
with st.expander("Learn more about each model"):
    st.write("**Faster R-CNN** is well-suited for general object detection and performs well with moderate-sized datasets. "
             "It’s effective for detecting bounding boxes around damaged areas with high accuracy.")
    st.write("**Mask R-CNN** builds on Faster R-CNN by adding instance segmentation. It’s ideal for situations where you need to "
             "both detect and segment damaged areas in high detail, providing pixel-level masks around damages.")
    st.write("**RetinaNet** is a single-stage detector known for speed and performance, particularly on small objects. "
             "It's suitable for real-time applications or scenarios where you want faster inference times, albeit with slightly less precision.")

# Model selection dropdown
model_choice = st.selectbox("Choose a model:", ["Faster R-CNN", "Mask R-CNN", "RetinaNet"])

# Confidence threshold slider
confidence_threshold = st.slider("Set confidence threshold:", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
predictor = load_predictor(model_choice, confidence_threshold)

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run prediction
    outputs = predictor(image_rgb)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores
    pred_classes = instances.pred_classes

    # Visualize predictions
    v = Visualizer(image_rgb, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(instances)

    # Display the annotated image
    annotated_image = Image.fromarray(out.get_image()[:, :, ::-1])
    st.image(annotated_image, caption=f"Annotated Image using {model_choice} with {confidence_threshold:.2f} Confidence", use_column_width=True)

    # Show damage classification results based on bounding box areas
    st.write("### Damage Detection Summary:")
    for i, box in enumerate(boxes):
        area = (box[2] - box[0]) * (box[3] - box[1])  # Calculate the area of the bounding box
        severity = classify_damage_severity(area)
        st.write(f"Damage {i + 1}: Confidence: {scores[i]:.2f}, Severity: {severity}, Area: {area:.2f}")

# Optional: Display memory usage (from tracemalloc)
st.write("Memory usage:")
st.write(tracemalloc.get_traced_memory())
