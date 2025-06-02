import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image # Pillow for image handling
import numpy as np
import cv2 # OpenCV (will be used by albumentations, and potentially for webcam if not using st.camera_input directly)

# --- Configuration ---
MODEL_PATH = "jaundice_mobilenetv3.pt"  # <<< IMPORTANT: Make sure this path is correct!
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406) # ImageNet means
STD = (0.229, 0.224, 0.225)  # ImageNet stds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Normal", "Jaundice"] # 0 for Normal, 1 for Jaundice

# --- Model Definition ---
# This must EXACTLY match the architecture of the model you trained and saved.
def get_model_architecture():
    # Using MobileNetV3 Small as in your training script
    model = models.mobilenet_v3_small(weights=None) # Load architecture, not pretrained weights
    # Modify the classifier head to output 1 logit (for binary classification)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model

# --- Load Model ---
# Use st.cache_resource to load the model only once and cache it.
@st.cache_resource
def load_trained_model(model_path):
    model = get_model_architecture()
    try:
        # Load the saved state dictionary.
        # map_location ensures it loads correctly whether you're on CPU or GPU.
        model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
        model.to(DEVICE)
        model.eval()  # Set the model to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please ensure the path is correct.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# --- Preprocessing ---
# Define the same transformations you used for validation/testing
def get_inference_transforms():
    return A.Compose([
        A.SmallestMaxSize(IMG_SIZE),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(MEAN, STD),
        ToTensorV2()
    ])

def preprocess_image_for_inference(image_pil):
    """
    Takes a PIL Image, converts to RGB, applies transformations, and prepares for model.
    """
    # Convert PIL image to NumPy array and ensure it's RGB
    img_np = np.array(image_pil.convert("RGB"))

    transforms = get_inference_transforms()
    augmented = transforms(image=img_np)
    img_tensor = augmented['image']

    # Add batch dimension (1, C, H, W) and send to device
    return img_tensor.unsqueeze(0).to(DEVICE)

# --- Prediction Function ---
def make_prediction(model, image_pil):
    if model is None:
        return None, None

    # Preprocess the image
    img_tensor = preprocess_image_for_inference(image_pil)

    # Make prediction
    with torch.no_grad(): # Disable gradient calculations
        logits = model(img_tensor)
        probability_jaundice = torch.sigmoid(logits).item() # Get the probability for the "Jaundice" class

    # Determine predicted class based on a 0.5 threshold
    predicted_class_idx = 1 if probability_jaundice > 0.5 else 0
    predicted_class_name = CLASS_NAMES[predicted_class_idx]

    return predicted_class_name, probability_jaundice

# --- Streamlit UI ---
st.set_page_config(page_title="Jaundice Detector", layout="centered")
st.title("Neonatal Jaundice Detector")
st.write(f"Utilizing model on device: **{DEVICE.upper()}**")
st.markdown("Upload an image of an infant's eye or use your webcam to check for potential signs of jaundice.")

# Load the model
model = load_trained_model(MODEL_PATH)

if model is None:
    st.warning("Model could not be loaded. Please check the console for errors and ensure the model path is correct.")
    st.stop() # Stop the app if model isn't loaded

# --- Input Method Selection ---
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose an image source:",
    ("Upload an Image", "Use Webcam")
)

# --- Prediction Display Area ---
prediction_placeholder = st.empty()

def display_prediction(predicted_class, probability):
    with prediction_placeholder.container():
        st.subheader("Prediction Result:")
        if predicted_class == "Jaundice":
            st.error(f"**{predicted_class} Detected** (Confidence: {probability:.2%})")
            st.warning("⚠️ This is a screening tool. Please consult a medical professional for an accurate diagnosis.")
        else: # Normal
            st.success(f"**{predicted_class}** (Confidence for Jaundice: {probability:.2%})")
            st.info("This suggests no immediate signs of jaundice based on the image.")


if input_method == "Upload an Image":
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file (jpg, png, jpeg):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

            if st.button("🔍 Analyze Uploaded Image"):
                with st.spinner("Analyzing..."):
                    predicted_class, probability = make_prediction(model, image)
                    if predicted_class is not None:
                        display_prediction(predicted_class, probability)
                    else:
                        prediction_placeholder.error("Could not make a prediction. Model might not be loaded correctly.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

elif input_method == "Use Webcam":
    st.header("📸 Use Webcam")
    # st.camera_input returns a BytesIO object when a picture is taken, or None
    img_file_buffer = st.camera_input("Take a picture (focus on the eye if possible):")

    if img_file_buffer is not None:
        try:
            # To read image file buffer with PIL:
            image = Image.open(img_file_buffer)
            st.image(image, caption="Captured Image.", use_column_width=True)

            if st.button("🔍 Analyze Webcam Image"):
                with st.spinner("Analyzing..."):
                    predicted_class, probability = make_prediction(model, image)
                    if predicted_class is not None:
                        display_prediction(predicted_class, probability)
                    else:
                        prediction_placeholder.error("Could not make a prediction. Model might not be loaded correctly.")
        except Exception as e:
            st.error(f"Error processing webcam image: {e}")

# --- Disclaimer ---
st.markdown("---")
# st.markdown("""
# **Disclaimer:** This application is a demonstration of a machine learning model
# and is **NOT a medical diagnostic tool**. The predictions made by this tool are
# not a substitute for professional medical advice, diagnosis, or treatment.
# Always seek the advice of a qualified health provider with any questions you may
# have regarding a medical condition.
# """)