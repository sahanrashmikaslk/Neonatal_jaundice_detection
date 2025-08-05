import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image # Pillow for image handling
import numpy as np
import cv2 # OpenCV for video capture and image manipulation

# --- Configuration (same as before) ---
MODEL_PATH = "jaundice_mobilenetv3_robust.pt"
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Normal", "Jaundice"]

# --- Model Definition (same as before) ---
def get_model_architecture():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model

# --- Load Model (modified to handle different model formats) ---
@st.cache_resource
def load_trained_model(model_path):
    model = get_model_architecture()
    try:
        state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
        
        # Check if this is the robust model format
        model_params = {}
        if isinstance(state_dict, dict) and 'base_model_state' in state_dict:
            st.info("Loading robust model with additional parameters")
            # Extract the actual model state dict
            model.load_state_dict(state_dict['base_model_state'])
            
            # Store the additional parameters
            if 'brightness_threshold' in state_dict:
                model_params['brightness_threshold'] = state_dict['brightness_threshold']
            if 'img_size' in state_dict:
                model_params['img_size'] = state_dict['img_size']
            if 'mean' in state_dict:
                model_params['mean'] = state_dict['mean']
            if 'std' in state_dict:
                model_params['std'] = state_dict['std']
        else:
            # Regular model format
            model.load_state_dict(state_dict)
            
        model.to(DEVICE)
        model.eval()
        return model, model_params
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please ensure the path is correct.")
        return None, {}
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, {}

# --- Preprocessing (same as before) ---
def get_inference_transforms():
    return A.Compose([
        A.SmallestMaxSize(IMG_SIZE),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(MEAN, STD),
        ToTensorV2()
    ])

def preprocess_frame_for_inference(frame_np_bgr):
    img_np_rgb = cv2.cvtColor(frame_np_bgr, cv2.COLOR_BGR2RGB) # OpenCV reads as BGR
    transforms = get_inference_transforms()
    augmented = transforms(image=img_np_rgb)
    img_tensor = augmented['image']
    return img_tensor.unsqueeze(0).to(DEVICE)

# --- Brightness Check Function ---
def check_image_brightness(image_np_bgr):
    """
    Calculate the average brightness of an image.
    Returns the brightness value (0-255) and a boolean indicating if it's too dark
    """
    # Convert to grayscale for brightness calculation
    gray = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2GRAY)
    # Calculate average brightness
    brightness = np.mean(gray)
    return brightness

# --- Prediction Function (modified to include confidence based on brightness) ---
def make_prediction_on_frame(model, frame_np_bgr, model_params):
    if model is None:
        return None, None, None, None
    
    # Check brightness
    brightness = check_image_brightness(frame_np_bgr)
    
    # Get brightness threshold from model parameters or use default
    brightness_threshold = model_params.get('brightness_threshold', 35)  # Default to 35 if not provided
    
    # Initialize confidence level (full confidence by default)
    confidence = 1.0
    
    # Define different brightness levels
    very_dark_threshold = brightness_threshold
    low_light_threshold = brightness_threshold * 1.5  # 50% brighter than too dark
    
    # If image is too dark, return early with "Too Dark" classification
    if brightness < very_dark_threshold:
        return "Too Dark", 0, brightness, 0.0
    
    # For low light conditions, we'll still make a prediction but with reduced confidence
    if brightness < low_light_threshold:
        confidence = 0.7  # Reduced confidence for low light conditions
    
    # Preprocess and get model prediction
    img_tensor = preprocess_frame_for_inference(frame_np_bgr)
    with torch.no_grad():
        logits = model(img_tensor)
        probability_jaundice = torch.sigmoid(logits).item()
    
    predicted_class_idx = 1 if probability_jaundice > 0.5 else 0
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    
    return predicted_class_name, probability_jaundice, brightness, confidence

# --- Streamlit UI ---
st.set_page_config(page_title="Jaundice Detector", layout="wide")
st.title("Neonatal Jaundice Detector")
st.write(f"Utilizing model on device: **{DEVICE.upper()}**")
st.markdown("Upload an image, use your webcam for a snapshot, or try the live feed detection.")

model, model_params = load_trained_model(MODEL_PATH)

if model is None:
    st.warning("Model could not be loaded. Please check the console for errors and ensure the model path is correct.")
    st.stop()

# Display model parameters if available
if model_params:
    with st.expander("Model Parameters"):
        for key, value in model_params.items():
            st.write(f"**{key}:** {value}")

if 'live_detection_active' not in st.session_state:
    st.session_state.live_detection_active = False
if 'webcam' not in st.session_state:
    st.session_state.webcam = None

st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose an image source:",
    ("Upload an Image", "Use Webcam Snapshot", "Live Feed Detection"),
    key="input_method_selector"
)

live_frame_placeholder = st.empty()
live_prediction_placeholder = st.empty()

def display_prediction_text(predicted_class, probability, brightness, confidence, placeholder):
    if predicted_class == "Too Dark":
        placeholder.warning(f"**Image Too Dark** (Brightness: {brightness:.2f}). Please use better lighting.")
    elif predicted_class == "Jaundice":
        if confidence < 1.0:
            placeholder.error(f"**{predicted_class} Detected** (Confidence: {probability:.2%}, Brightness: {brightness:.2f})\n⚠️ **Reliability: {confidence:.2%}** - Low light may affect accuracy")
        else:
            placeholder.error(f"**{predicted_class} Detected** (Confidence: {probability:.2%}, Brightness: {brightness:.2f})")
    else: # Normal
        if confidence < 1.0:
            placeholder.success(f"**{predicted_class}** (Confidence for Jaundice: {probability:.2%}, Brightness: {brightness:.2f})\n⚠️ **Reliability: {confidence:.2%}** - Low light may affect accuracy")
        else:
            placeholder.success(f"**{predicted_class}** (Confidence for Jaundice: {probability:.2%}, Brightness: {brightness:.2f})")

if input_method == "Upload an Image":
    st.session_state.live_detection_active = False
    if st.session_state.webcam is not None:
        st.session_state.webcam.release()
        st.session_state.webcam = None
    live_frame_placeholder.empty()
    live_prediction_placeholder.empty()

    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file (jpg, png, jpeg):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True) # CORRECTED
        if st.button("🔍 Analyze Uploaded Image"):
            with st.spinner("Analyzing..."):
                img_np = np.array(image.convert("RGB"))
                img_np_bgr_for_func = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                predicted_class, probability, brightness, confidence = make_prediction_on_frame(model, img_np_bgr_for_func, model_params)
                if predicted_class is not None:
                    display_prediction_text(predicted_class, probability, brightness, confidence, live_prediction_placeholder)
                else:
                    live_prediction_placeholder.error("Could not make a prediction.")

elif input_method == "Use Webcam Snapshot":
    st.session_state.live_detection_active = False
    if st.session_state.webcam is not None:
        st.session_state.webcam.release()
        st.session_state.webcam = None
    live_frame_placeholder.empty()
    live_prediction_placeholder.empty()

    st.header("📸 Use Webcam Snapshot")
    img_file_buffer = st.camera_input("Take a picture (focus on the eye if possible):")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Captured Image.", use_container_width=True) # CORRECTED
        if st.button("🔍 Analyze Webcam Snapshot"):
            with st.spinner("Analyzing..."):
                img_np = np.array(image.convert("RGB"))
                img_np_bgr_for_func = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                predicted_class, probability, brightness, confidence = make_prediction_on_frame(model, img_np_bgr_for_func, model_params)
                if predicted_class is not None:
                    display_prediction_text(predicted_class, probability, brightness, confidence, live_prediction_placeholder)
                else:
                    live_prediction_placeholder.error("Could not make a prediction.")

elif input_method == "Live Feed Detection":
    st.header("📹 Live Feed Jaundice Detection")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Live Detection", key="start_live"):
            if not st.session_state.live_detection_active:
                st.session_state.live_detection_active = True
                st.session_state.webcam = cv2.VideoCapture(0)
                if not st.session_state.webcam.isOpened():
                    st.error("Could not open webcam. Please check permissions or if another app is using it.")
                    st.session_state.live_detection_active = False
                    st.session_state.webcam = None
                else:
                    st.rerun()

    with col2:
        if st.button("🛑 Stop Live Detection", key="stop_live"):
            st.session_state.live_detection_active = False
            if st.session_state.webcam is not None:
                st.session_state.webcam.release()
                st.session_state.webcam = None
            live_frame_placeholder.empty()
            live_prediction_placeholder.empty()
            st.rerun()

    if st.session_state.live_detection_active and st.session_state.webcam is not None and st.session_state.webcam.isOpened():
        live_prediction_placeholder.info("Live detection active... Point camera at the subject's eyes.")
        while st.session_state.live_detection_active:
            ret, frame = st.session_state.webcam.read()
            if not ret:
                live_prediction_placeholder.error("Failed to grab frame from webcam. Stopping.")
                st.session_state.live_detection_active = False
                if st.session_state.webcam is not None: st.session_state.webcam.release()
                st.session_state.webcam = None
                st.rerun()
                break

            predicted_class, probability, brightness, confidence = make_prediction_on_frame(model, frame, model_params)
            
            # Determine text color based on prediction
            if predicted_class == "Too Dark":
                color = (0, 165, 255)  # Orange in BGR
                display_text = f"Too Dark (Brightness: {brightness:.2f})"
            elif predicted_class == "Jaundice":
                color = (0, 0, 255)  # Red in BGR
                display_text = f"{predicted_class} ({probability:.2%})"
            else:  # Normal
                color = (0, 255, 0)  # Green in BGR
                display_text = f"{predicted_class} ({probability:.2%})"
                
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add brightness indicator
            cv2.putText(frame, f"Brightness: {brightness:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add reliability indicator based on confidence
            if confidence < 1.0:
                reliability_text = f"Reliability: {confidence:.2%} (Low light)"
                cv2.putText(frame, reliability_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            
            live_frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True) # CORRECTED

    elif st.session_state.live_detection_active and (st.session_state.webcam is None or not st.session_state.webcam.isOpened()):
        live_prediction_placeholder.error("Webcam is not available for live detection. Try starting again or check camera.")
        st.session_state.live_detection_active = False

st.markdown("---")
