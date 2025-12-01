import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import albumentations as A
from typing import Optional
from albumentations.pytorch import ToTensorV2
from PIL import Image  # Pillow for image handling
import numpy as np
import cv2 # OpenCV for video capture and image manipulation
from baby_presence_detector import BabyPresenceDetector, PresenceResult

# --- Configuration ---
MODEL_PATH = "jaundice_mobilenetv3_v4.pt"
CONFIG_PATH = "jaundice_mobilenetv3_v4_config.pt"
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Normal", "Jaundice"]
QUALITY_THRESHOLD = 0.3  # Below this, image quality is too poor

# --- Dual-Head Model Definition ---
class JaundiceQualityModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # Use MobileNetV3 backbone up to the last hidden layer
        self.backbone = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            nn.Flatten(),
            base_model.classifier[0],  # Linear to 1024
            base_model.classifier[1],  # Hardswish
            base_model.classifier[2],  # Dropout
        )
        in_features = base_model.classifier[3].in_features

        # Two heads: one for jaundice, one for quality
        self.jaundice_head = nn.Linear(in_features, 1)
        self.quality_head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        logits_j = self.jaundice_head(x)
        logits_q = self.quality_head(x)
        return logits_j, logits_q

def get_model_architecture():
    base = models.mobilenet_v3_small(weights=None)
    model = JaundiceQualityModel(base)
    return model

# --- Load Dual-Head Model ---
@st.cache_resource
def load_trained_model(model_path, config_path):
    model = get_model_architecture()
    model_params = {}
    
    try:
        # Load model weights
        state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
        model.load_state_dict(state_dict)
        st.success("✅ Dual-head model loaded successfully (Jaundice + Quality heads)")
        
        # Load config if available
        try:
            config = torch.load(config_path, map_location=torch.device(DEVICE))
            model_params = config
            st.info(f"📋 Model config loaded: IMG_SIZE={config.get('img_size')}, Quality Threshold={config.get('quality_threshold')}")
        except:
            st.warning("Config file not found, using default parameters")
            model_params = {
                'img_size': IMG_SIZE,
                'mean': MEAN,
                'std': STD,
                'quality_threshold': QUALITY_THRESHOLD
            }
            
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

# --- Presence detector (cached so it loads once) ---
@st.cache_resource
def get_presence_detector():
    return BabyPresenceDetector()

# --- Dual-Head Prediction Function ---
def make_prediction_on_frame(model, frame_np_bgr, model_params, presence_detector=None):
    if model is None:
        return None, None, None, None, None, None
    
    # Calculate brightness for reference
    brightness = check_image_brightness(frame_np_bgr)
    
    # Check for baby presence first
    presence_result: Optional[PresenceResult] = None
    if presence_detector is not None:
        presence_result = presence_detector.is_baby_present(frame_np_bgr)
        if not presence_result.is_present:
            return "No Baby Detected", 0.0, 0.0, brightness, presence_result, "presence_check"

    # Preprocess and get dual-head model predictions
    img_tensor = preprocess_frame_for_inference(frame_np_bgr)
    with torch.no_grad():
        logits_j, logits_q = model(img_tensor)  # Dual outputs
        probability_jaundice = torch.sigmoid(logits_j).item()
        quality_score = torch.sigmoid(logits_q).item()
    
    # Get quality threshold from model parameters
    quality_threshold = model_params.get('quality_threshold', QUALITY_THRESHOLD)
    
    # Check if quality is too poor for reliable diagnosis
    if quality_score < quality_threshold:
        return "Poor Quality", probability_jaundice, quality_score, brightness, presence_result, "low_quality"
    
    # Make jaundice prediction
    predicted_class_idx = 1 if probability_jaundice > 0.5 else 0
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    
    return predicted_class_name, probability_jaundice, quality_score, brightness, presence_result, "success"

# --- Streamlit UI ---
st.set_page_config(page_title="Dual-Head Jaundice Detector", layout="wide", page_icon="👶")
st.title("👶 Neonatal Jaundice Detector (Dual-Head AI)")
st.markdown("### Multi-Task Learning: Jaundice Detection + Quality Assessment")
st.write(f"🖥️ Running on: **{DEVICE.upper()}**")
st.markdown("---")
st.info("📊 This model uses **dual-head architecture** to simultaneously predict:\n\n"
        "1️⃣ **Jaundice Detection** - Normal vs Jaundice classification\n\n"
        "2️⃣ **Image Quality** - Learned assessment of image suitability\n\n"
        "💡 Quality scores inform the reliability of jaundice predictions.")

model, model_params = load_trained_model(MODEL_PATH, CONFIG_PATH)
presence_detector = get_presence_detector()

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
if 'selected_camera' not in st.session_state:
    st.session_state.selected_camera = 0

# --- Helper function to detect available cameras ---
def get_available_cameras(max_cameras=5):
    """Detect available cameras by trying to open them"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose an image source:",
    ("Upload an Image", "Use Webcam Snapshot", "Live Feed Detection"),
    key="input_method_selector"
)

# Camera selection in sidebar
if input_method in ["Use Webcam Snapshot", "Live Feed Detection"]:
    st.sidebar.header("Camera Settings")
    
    # Only detect cameras if not actively using one
    if not st.session_state.live_detection_active:
        available_cameras = get_available_cameras()
        
        if available_cameras:
            camera_labels = {cam: f"Camera {cam}" for cam in available_cameras}
            # Try to get camera names (Windows-specific)
            try:
                import subprocess
                result = subprocess.run(['powershell', '-Command', 
                                       'Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName'],
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    names = [n.strip() for n in result.stdout.strip().split('\n') if n.strip()]
                    for idx, name in enumerate(names[:len(available_cameras)]):
                        if idx < len(available_cameras):
                            camera_labels[available_cameras[idx]] = f"Camera {available_cameras[idx]}: {name}"
            except:
                pass
            
            selected_camera = st.sidebar.selectbox(
                "Select Camera:",
                options=available_cameras,
                format_func=lambda x: camera_labels.get(x, f"Camera {x}"),
                index=available_cameras.index(st.session_state.selected_camera) if st.session_state.selected_camera in available_cameras else 0,
                key="camera_selector"
            )
            st.session_state.selected_camera = selected_camera
            st.sidebar.info(f"🎥 Using: {camera_labels.get(selected_camera, f'Camera {selected_camera}')}")
        else:
            st.sidebar.error("❌ No cameras detected!")
            st.session_state.selected_camera = 0
    else:
        # Show camera info when live detection is active
        st.sidebar.info(f"🎥 Active: Camera {st.session_state.selected_camera}")
        st.sidebar.warning("⚠️ Stop live detection to change camera")

live_frame_placeholder = st.empty()
live_prediction_placeholder = st.empty()

def display_prediction_text(predicted_class, probability, quality_score, brightness, placeholder, presence_result: Optional[PresenceResult], status):
    # Display based on status and quality
    if status == "presence_check":
        reason = f" ({presence_result.reason})" if presence_result and getattr(presence_result, "reason", None) else ""
        placeholder.warning(f"⚠️ **No baby detected**{reason}. Skipping jaundice analysis.")
    elif status == "low_quality":
        placeholder.error(f"❌ **Image Quality Too Poor** (Quality Score: {quality_score:.2%})\n\n"
                         f"The model cannot provide reliable diagnosis. Please ensure:\n"
                         f"• Better lighting (current brightness: {brightness:.2f})\n"
                         f"• Clear focus on the eye\n"
                         f"• Minimal motion blur")
    elif predicted_class == "Jaundice":
        if quality_score >= 0.7:
            placeholder.error(f"🔴 **Jaundice Detected**\n\n"
                            f"• Jaundice Probability: **{probability:.2%}**\n"
                            f"• Image Quality: **{quality_score:.2%}** (High confidence)\n"
                            f"• Brightness: {brightness:.2f}\n\n"
                            f"⚕️ *Consult a healthcare professional for proper diagnosis.*")
        else:
            placeholder.warning(f"⚠️ **Jaundice Detected** (Medium Confidence)\n\n"
                              f"• Jaundice Probability: **{probability:.2%}**\n"
                              f"• Image Quality: **{quality_score:.2%}** (Medium confidence)\n"
                              f"• Brightness: {brightness:.2f}\n\n"
                              f"⚕️ *Result may be less reliable due to image quality. Consult a healthcare professional.*")
    else:  # Normal
        if quality_score >= 0.7:
            placeholder.success(f"✅ **Normal - No Jaundice Detected**\n\n"
                              f"• Jaundice Probability: **{probability:.2%}**\n"
                              f"• Image Quality: **{quality_score:.2%}** (High confidence)\n"
                              f"• Brightness: {brightness:.2f}")
        else:
            placeholder.info(f"ℹ️ **Normal** (Medium Confidence)\n\n"
                           f"• Jaundice Probability: **{probability:.2%}**\n"
                           f"• Image Quality: **{quality_score:.2%}** (Medium confidence)\n"
                           f"• Brightness: {brightness:.2f}\n\n"
                           f"💡 *Consider retaking with better lighting for higher confidence.*")

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
                predicted_class, probability, quality_score, brightness, presence_result, status = make_prediction_on_frame(model, img_np_bgr_for_func, model_params, presence_detector)
                if predicted_class is not None:
                    display_prediction_text(predicted_class, probability, quality_score, brightness, live_prediction_placeholder, presence_result, status)
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
                predicted_class, probability, quality_score, brightness, presence_result, status = make_prediction_on_frame(model, img_np_bgr_for_func, model_params, presence_detector)
                if predicted_class is not None:
                    display_prediction_text(predicted_class, probability, quality_score, brightness, live_prediction_placeholder, presence_result, status)
                else:
                    live_prediction_placeholder.error("Could not make a prediction.")

elif input_method == "Live Feed Detection":
    st.header("📹 Live Feed Jaundice Detection")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Live Detection", key="start_live"):
            if not st.session_state.live_detection_active:
                st.session_state.live_detection_active = True
                st.session_state.webcam = cv2.VideoCapture(st.session_state.selected_camera)
                if not st.session_state.webcam.isOpened():
                    st.error(f"Could not open Camera {st.session_state.selected_camera}. Please check permissions or try a different camera.")
                    st.session_state.live_detection_active = False
                    st.session_state.webcam = None
                else:
                    st.success(f"✅ Camera {st.session_state.selected_camera} opened successfully!")
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

            predicted_class, probability, quality_score, brightness, presence_result, status = make_prediction_on_frame(model, frame, model_params, presence_detector)
            
            # Determine text color and display based on status
            if status == "presence_check":
                color = (0, 165, 255)  # Orange in BGR
                reason = f" ({presence_result.reason})" if presence_result and getattr(presence_result, "reason", None) else ""
                display_text = f"No Baby{reason}"
            elif status == "low_quality":
                color = (0, 0, 255)  # Red in BGR
                display_text = f"Poor Quality (Q: {quality_score:.2%})"
            elif predicted_class == "Jaundice":
                if quality_score >= 0.7:
                    color = (0, 0, 255)  # Red in BGR
                    display_text = f"JAUNDICE ({probability:.2%})"
                else:
                    color = (0, 165, 255)  # Orange in BGR
                    display_text = f"Jaundice? ({probability:.2%})"
            else:  # Normal
                if quality_score >= 0.7:
                    color = (0, 255, 0)  # Green in BGR
                    display_text = f"Normal ({probability:.2%})"
                else:
                    color = (255, 255, 0)  # Cyan in BGR
                    display_text = f"Normal? ({probability:.2%})"
                
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add quality score indicator with color coding
            quality_color = (0, 255, 0) if quality_score >= 0.7 else (0, 165, 255) if quality_score >= 0.3 else (0, 0, 255)
            cv2.putText(frame, f"Quality: {quality_score:.2%}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
            
            # Add brightness indicator
            cv2.putText(frame, f"Brightness: {brightness:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add quality bar visualization
            bar_width = int(quality_score * 200)
            cv2.rectangle(frame, (10, 100), (210, 120), (50, 50, 50), -1)  # Background
            cv2.rectangle(frame, (10, 100), (10 + bar_width, 120), quality_color, -1)  # Quality bar
            
            live_frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True) # CORRECTED

    elif st.session_state.live_detection_active and (st.session_state.webcam is None or not st.session_state.webcam.isOpened()):
        live_prediction_placeholder.error("Webcam is not available for live detection. Try starting again or check camera.")
        st.session_state.live_detection_active = False

st.markdown("---")
