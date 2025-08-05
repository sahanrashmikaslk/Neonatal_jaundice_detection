import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image # Pillow for image handling
import numpy as np
import cv2 # OpenCV for video capture and image manipulation
import time # For time-related functions

# --- Configuration (updated to use robust model) ---
MODEL_PATH = "jaundice_mobilenetv3_robust.pt"  # Changed to use robust model
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Normal", "Jaundice"]
BRIGHTNESS_THRESHOLD = 70  # For dark image detection

# --- Model Definition (updated for robust model) ---
def get_model_architecture():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model

# --- Lighting Robust Model Wrapper ---
class LightingRobustJaundiceModel:
    """
    A wrapper model that handles dark lighting conditions to prevent false positives.
    It detects dark images and either enhances them or reduces confidence in predictions.
    """
    def __init__(self, base_model, device="cpu", brightness_threshold=70):
        self.base_model = base_model
        self.device = device
        self.brightness_threshold = brightness_threshold
        self.base_model.eval()
    
    def enhance_image(self, image):
        """Enhance a dark image to improve visibility"""
        # Convert to LAB color space for better brightness adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Split channels
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        # Merge channels
        lab = cv2.merge((l, a, b))
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def is_dark_image(self, image):
        """Check if image is too dark based on average brightness"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        return avg_brightness < self.brightness_threshold
    
    def predict(self, image_tensor):
        """Make prediction using the base model"""
        with torch.no_grad():
            logits = self.base_model(image_tensor)
            prob = torch.sigmoid(logits).item()
        return prob

# --- Load Model (updated for robust model) ---
@st.cache_resource
def load_trained_model(model_path):
    try:
        # Load the dictionary with all components
        saved_dict = torch.load(model_path, map_location=torch.device(DEVICE))
        
        # Create base model
        model = get_model_architecture()
        
        # Load weights
        model.load_state_dict(saved_dict['base_model_state'])
        model.to(DEVICE)
        
        # Create robust model with saved parameters
        brightness_threshold = saved_dict.get('brightness_threshold', BRIGHTNESS_THRESHOLD)
        robust_model = LightingRobustJaundiceModel(
            base_model=model,
            device=DEVICE,
            brightness_threshold=brightness_threshold
        )
        
        return robust_model
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please ensure the path is correct.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# --- Preprocessing (updated to use PyTorch transforms with explicit type handling) ---
def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

def preprocess_frame_for_inference(frame_np_bgr):
    # Check and convert input if necessary
    if frame_np_bgr is None:
        raise ValueError("Input frame is None")
    
    # Handle different data types - ensure we have uint8 BGR image
    if not isinstance(frame_np_bgr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(frame_np_bgr)}")
    
    # Check shape and channels
    if len(frame_np_bgr.shape) != 3 or frame_np_bgr.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {frame_np_bgr.shape}")
    
    # Ensure data type is uint8
    if frame_np_bgr.dtype != np.uint8:
        # Try to convert
        try:
            frame_np_bgr = frame_np_bgr.astype(np.uint8)
        except Exception as e:
            raise TypeError(f"Cannot convert frame to uint8: {e}")
    
    # Now proceed with normal processing
    img_np_rgb = cv2.cvtColor(frame_np_bgr, cv2.COLOR_BGR2RGB) # OpenCV reads as BGR
    
    # Convert to PIL Image with explicit uint8 type
    img_np_rgb = img_np_rgb.astype(np.uint8)
    pil_image = Image.fromarray(img_np_rgb)
    
    # Manual conversion to tensor to avoid compatibility issues
    img_tensor = transforms.ToTensor()(pil_image)
    # Apply normalization separately
    img_tensor = transforms.Normalize(MEAN, STD)(img_tensor)
    
    return img_tensor.unsqueeze(0).to(DEVICE), img_np_rgb

# --- Prediction Function (updated for robust model with improved error handling) ---
def make_prediction_on_frame(robust_model, frame_np_bgr):
    if robust_model is None:
        return None, None, None, None
    
    try:
        # Convert BGR to RGB and preprocess with added error handling
        img_tensor, img_np_rgb = preprocess_frame_for_inference(frame_np_bgr)
        
        # Check if the image is dark
        is_dark = robust_model.is_dark_image(img_np_rgb)
        
        # Enhance dark image if needed
        enhanced_image = None
        if is_dark:
            enhanced_image = robust_model.enhance_image(img_np_rgb)
            # Also preprocess the enhanced image with explicit type handling
            enhanced_image = enhanced_image.astype(np.uint8)
            pil_enhanced = Image.fromarray(enhanced_image)
            
            # Manual tensor conversion and normalization
            enhanced_tensor = transforms.ToTensor()(pil_enhanced)
            enhanced_tensor = transforms.Normalize(MEAN, STD)(enhanced_tensor)
            
            probability_jaundice = robust_model.predict(enhanced_tensor.unsqueeze(0).to(DEVICE))
        else:
            probability_jaundice = robust_model.predict(img_tensor)
        
        # Calculate confidence based on lighting
        confidence = 1.0 if not is_dark else 0.7
        
        # Determine predicted class
        predicted_class_idx = 1 if probability_jaundice > 0.5 else 0
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        
        return predicted_class_name, probability_jaundice, confidence, is_dark
    
    except Exception as e:
        # Provide more specific error information for debugging
        error_msg = str(e)
        if "data type" in error_msg and "|O" in error_msg:
            # This is the specific error we're trying to handle
            raise TypeError(f"Cannot handle object data type: {error_msg}. The frame has an incompatible format.")
        else:
            # Re-raise the original exception
            raise

# --- Streamlit UI ---
st.set_page_config(page_title="Jaundice Detector", layout="wide")
st.title("Neonatal Jaundice Detector with Lighting Robustness")
st.write(f"Utilizing robust model on device: **{DEVICE.upper()}**")
st.markdown("Upload an image, use your webcam for a snapshot, or try the live feed detection. This model includes lighting robustness to handle poor lighting conditions.")

model = load_trained_model(MODEL_PATH)

if model is None:
    st.warning("Model could not be loaded. Please check the console for errors and ensure the model path is correct.")
    st.stop()

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

# Add camera selector in sidebar
if input_method == "Live Feed Detection" or input_method == "Use Webcam Snapshot":
    st.sidebar.subheader("Camera Settings")
    camera_index = st.sidebar.selectbox(
        "Select camera (if you have multiple):",
        options=[0, 1, 2],
        index=0,
        help="If you have multiple cameras, you can select which one to use here. Try different options if your camera isn't working."
    )
    
    # Force reset option
    if st.sidebar.button("Force Reset Camera"):
        if st.session_state.get('webcam') is not None:
            st.session_state.webcam.release()
            st.session_state.webcam = None
        st.session_state.live_detection_active = False
        st.sidebar.success("Camera has been reset. Please try again.")
        st.rerun()
    
    # Store the camera index in session state
    if 'camera_index' not in st.session_state or st.session_state.camera_index != camera_index:
        st.session_state.camera_index = camera_index
        # If we changed the camera and are in live mode, we need to restart the camera
        if st.session_state.get('live_detection_active', False) and st.session_state.get('webcam') is not None:
            st.session_state.webcam.release()
            st.session_state.webcam = None
            st.rerun()

live_frame_placeholder = st.empty()
live_prediction_placeholder = st.empty()

def display_prediction_text(predicted_class, probability, confidence, is_dark, placeholder):
    if is_dark:
        placeholder.warning("⚠️ **Low Light Detected** - Image quality may affect results")
    
    if predicted_class == "Jaundice":
        placeholder.error(f"**{predicted_class} Detected** (Probability: {probability:.2%}, Confidence: {confidence:.2%})")
    else: # Normal
        placeholder.success(f"**{predicted_class}** (Jaundice Probability: {probability:.2%}, Confidence: {confidence:.2%})")
        
    if is_dark:
        placeholder.info("Image has been enhanced to improve detection in low light")

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
        st.image(image, caption="Uploaded Image.", use_container_width=True)
        if st.button("🔍 Analyze Uploaded Image"):
            with st.spinner("Analyzing..."):
                img_np = np.array(image.convert("RGB"))
                img_np_bgr_for_func = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                predicted_class, probability, confidence, is_dark = make_prediction_on_frame(model, img_np_bgr_for_func)
                if predicted_class is not None:
                    display_prediction_text(predicted_class, probability, confidence, is_dark, live_prediction_placeholder)
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
    camera_instructions = f"Using camera index: {st.session_state.get('camera_index', 0)}. If camera isn't working, try changing the camera in the sidebar."
    st.info(camera_instructions)
    img_file_buffer = st.camera_input("Take a picture (focus on the eye if possible):")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Captured Image.", use_container_width=True)
        if st.button("🔍 Analyze Webcam Snapshot"):
            with st.spinner("Analyzing..."):
                img_np = np.array(image.convert("RGB"))
                img_np_bgr_for_func = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                predicted_class, probability, confidence, is_dark = make_prediction_on_frame(model, img_np_bgr_for_func)
                if predicted_class is not None:
                    display_prediction_text(predicted_class, probability, confidence, is_dark, live_prediction_placeholder)
                else:
                    live_prediction_placeholder.error("Could not make a prediction.")

elif input_method == "Live Feed Detection":
    st.header("📹 Live Feed Jaundice Detection")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Live Detection", key="start_live"):
            if not st.session_state.live_detection_active:
                st.session_state.live_detection_active = True
                # Try different backend options for webcam access
                try:
                    # Get the selected camera index
                    camera_idx = st.session_state.get('camera_index', 0)
                    
                    # Try with DirectShow backend first (often works best on Windows)
                    st.info(f"Attempting to access camera {camera_idx}...")
                    st.session_state.webcam = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
                    
                    # If that fails, try with other backends
                    if not st.session_state.webcam.isOpened():
                        st.info(f"Trying default backend for camera {camera_idx}...")
                        st.session_state.webcam = cv2.VideoCapture(camera_idx)
                    
                    # If still not working, try alternative camera index as fallback
                    if not st.session_state.webcam.isOpened() and camera_idx == 0:
                        st.info("Trying fallback camera...")
                        st.session_state.webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                    
                    if not st.session_state.webcam.isOpened():
                        st.error(f"Could not open camera {camera_idx}. Please check permissions or if another app is using it.")
                        st.session_state.live_detection_active = False
                        st.session_state.webcam = None
                    else:
                        # Set properties for better performance
                        st.session_state.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        st.session_state.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        st.session_state.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        st.session_state.webcam.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better stability
                        
                        # Add a small delay to let the camera initialize
                        time.sleep(0.5)
                        
                        # Read a couple of frames to "warm up" the camera
                        for _ in range(5):
                            st.session_state.webcam.read()
                            time.sleep(0.1)
                            
                        st.success(f"Camera {camera_idx} initialized successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error accessing webcam: {str(e)}")
                    st.session_state.live_detection_active = False
                    st.session_state.webcam = None

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
        
        # Add frame retry counter
        retry_count = 0
        max_retries = 5
        
        while st.session_state.live_detection_active:
            # Try to read a frame
            ret, frame = st.session_state.webcam.read()
            
            # Handle frame capture failure
            if not ret:
                retry_count += 1
                if retry_count >= max_retries:
                    live_prediction_placeholder.error("Failed to grab frames from webcam after multiple attempts. Stopping.")
                    st.session_state.live_detection_active = False
                    if st.session_state.webcam is not None: 
                        st.session_state.webcam.release()
                    st.session_state.webcam = None
                    st.rerun()
                    break
                else:
                    # Short pause before retry
                    time.sleep(0.1)
                    continue  # Skip this iteration and try again
            
            # Reset retry counter if we got a frame
            retry_count = 0
            
            # Process the frame
            try:
                # Check frame data type and shape
                if frame is None:
                    raise ValueError("Frame is None")
                
                # Add extra checks for data type issues
                frame_type = frame.dtype
                frame_shape = frame.shape
                
                # Log frame information for debugging
                if retry_count == 0:  # Only log once to avoid spamming
                    st.session_state['frame_debug_info'] = f"Frame shape: {frame_shape}, dtype: {frame_type}"
                
                # For certain problematic data types, try to convert
                if frame_type != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Handle unusual shapes
                if len(frame_shape) != 3 or frame_shape[2] != 3:
                    raise ValueError(f"Invalid frame shape: {frame_shape}. Expected 3-channel image.")
                
                # Now proceed with prediction
                predicted_class, probability, confidence, is_dark = make_prediction_on_frame(model, frame)
                
                # Add visual indicators for lighting and confidence
                display_frame = frame.copy()
                
                # Display text with result
                display_text = f"{predicted_class} ({probability:.2%})"
                color = (0, 0, 255) if predicted_class == "Jaundice" else (0, 255, 0) # BGR
                cv2.putText(display_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Add lighting indicator
                if is_dark:
                    cv2.putText(display_frame, "LOW LIGHT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Display the frame
                live_frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                # Reset any error warnings
                if 'error_displayed' in st.session_state and st.session_state['error_displayed']:
                    st.session_state['error_displayed'] = False
                    live_prediction_placeholder.info("Live detection active... Point camera at the subject's eyes.")
            
            except TypeError as e:
                # Handle type errors specifically
                if "|O" in str(e) or "data type" in str(e):
                    if not st.session_state.get('error_displayed', False):
                        live_prediction_placeholder.warning(f"Camera returned incompatible data format. Trying to recover...")
                        st.session_state['error_displayed'] = True
                    # Just skip this frame and try again
                    time.sleep(0.2)
                    continue
                else:
                    live_prediction_placeholder.warning(f"Frame processing error: {str(e)}. Continuing...")
                    continue
            
            except Exception as e:
                live_prediction_placeholder.warning(f"Frame processing error: {str(e)}. Continuing...")
                continue

    elif st.session_state.live_detection_active and (st.session_state.webcam is None or not st.session_state.webcam.isOpened()):
        live_prediction_placeholder.error("Webcam is not available for live detection. Try starting again or check camera.")
        st.session_state.live_detection_active = False

st.markdown("---")

# To run the app, use the command: streamlit run app.py
