#!/usr/bin/env python3
"""
Jaundice Detection Server for Pi Monitoring System
FastAPI server that provides jaundice detection from camera stream
With automatic detection every 10 minutes and ThingsBoard integration
"""

import torch
import torchvision.models as models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import logging
import asyncio
import json
import os
import paho.mqtt.client as mqtt
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_PATH = "/home/sahan/jaundice_detection/jaundice_mobilenetv3_robust.pt"
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = "cpu"  # Pi doesn't have CUDA
CLASS_NAMES = ["Normal", "Jaundice"]
INFANT_CAMERA_URL = "http://localhost:8081/?action=stream"  # V380 camera for infant

# Auto-detection configuration
AUTO_DETECT_INTERVAL = 600  # 10 minutes in seconds

# ThingsBoard configuration
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'incubator_monitoring_with_thingsboard_integration', 'config')
CONFIG_PATH = os.path.join(CONFIG_DIR, 'device_credentials.json')

# Load ThingsBoard credentials
try:
    with open(CONFIG_PATH, 'r') as f:
        device_config = json.load(f)
    TB_HOST = device_config['thingsboard_host']
    TB_PORT = device_config['mqtt_port']
    ACCESS_TOKEN = device_config['access_token']
    logger.info("✓ ThingsBoard configuration loaded")
except Exception as e:
    logger.warning(f"Could not load ThingsBoard config: {e}. Running without TB integration.")
    TB_HOST = None
    TB_PORT = None
    ACCESS_TOKEN = None

app = FastAPI(title="Jaundice Detection API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_params = {}
last_detection_result = None
auto_detection_task = None
tb_client = None

# --- Model Definition ---
def get_model_architecture():
    """Create MobileNetV3 model architecture"""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model

# --- ThingsBoard Client ---
class ThingsBoardClient:
    """MQTT client for ThingsBoard communication"""
    
    def __init__(self):
        if not ACCESS_TOKEN:
            logger.warning("ThingsBoard not configured")
            self.enabled = False
            return
            
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.client.username_pw_set(ACCESS_TOKEN)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.connected = False
        self.enabled = True
        self.telemetry_topic = 'v1/devices/me/telemetry'
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("✓ Connected to ThingsBoard successfully")
            self.connected = True
        else:
            logger.error(f"✗ ThingsBoard connection failed with code {rc}")
            self.connected = False
    
    def on_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from ThingsBoard (code: {rc})")
        self.connected = False
        
    def connect(self):
        """Connect to ThingsBoard MQTT broker"""
        if not self.enabled:
            return False
            
        try:
            self.client.connect(TB_HOST, TB_PORT, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.5)
            
            if not self.connected:
                raise Exception("Connection timeout")
                
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ThingsBoard: {e}")
            return False
    
    def publish_jaundice_data(self, detection_result):
        """Publish jaundice detection data to ThingsBoard"""
        if not self.enabled:
            return False
        
        # Reconnect if not connected
        if not self.connected:
            logger.warning("Not connected to ThingsBoard, attempting to reconnect...")
            if not self.connect():
                logger.error("Failed to reconnect to ThingsBoard")
                return False
        
        try:
            # Prepare telemetry data
            telemetry = {
                'jaundice_detected': detection_result.get('jaundice_detected', False),
                'jaundice_confidence': round(detection_result.get('confidence', 0) * 100, 2),
                'jaundice_probability': round(detection_result.get('probability', 0) * 100, 2),
                'jaundice_brightness': round(detection_result.get('brightness', 0), 2),
                'jaundice_status': detection_result.get('predicted_class', 'Unknown'),
                'jaundice_reliability': round(detection_result.get('reliability', 1.0) * 100, 2),
                'timestamp': int(time.time() * 1000)
            }
            
            payload = json.dumps(telemetry)
            result = self.client.publish(self.telemetry_topic, payload, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"✓ Jaundice data published to ThingsBoard: {telemetry}")
                return True
            else:
                logger.error(f"✗ Publish failed with code {result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing to ThingsBoard: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ThingsBoard"""
        if self.enabled:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Disconnected from ThingsBoard")

# --- Load Model ---
def load_trained_model(model_path):
    """Load the trained jaundice detection model"""
    try:
        model = get_model_architecture()
        state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
        
        # Check if this is the robust model format
        params = {}
        if isinstance(state_dict, dict) and 'base_model_state' in state_dict:
            logger.info("Loading robust model with additional parameters")
            model.load_state_dict(state_dict['base_model_state'])
            
            # Store the additional parameters
            if 'brightness_threshold' in state_dict:
                params['brightness_threshold'] = state_dict['brightness_threshold']
            if 'img_size' in state_dict:
                params['img_size'] = state_dict['img_size']
            if 'mean' in state_dict:
                params['mean'] = state_dict['mean']
            if 'std' in state_dict:
                params['std'] = state_dict['std']
        else:
            # Regular model format
            model.load_state_dict(state_dict)
            
        model.to(DEVICE)
        model.eval()
        logger.info(f"✓ Model loaded successfully from {model_path}")
        return model, params
    except FileNotFoundError:
        logger.error(f"Model file not found at '{model_path}'")
        return None, {}
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        return None, {}

# --- Preprocessing ---
def get_inference_transforms():
    """Get image preprocessing transforms"""
    return A.Compose([
        A.SmallestMaxSize(IMG_SIZE),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(MEAN, STD),
        ToTensorV2()
    ])

def preprocess_frame(frame_bgr):
    """Preprocess frame for inference"""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    transforms = get_inference_transforms()
    augmented = transforms(image=img_rgb)
    img_tensor = augmented['image']
    return img_tensor.unsqueeze(0).to(DEVICE)

# --- Brightness Check ---
def check_image_brightness(image_bgr):
    """Calculate the average brightness of an image (0-255)"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness

# --- Prediction Function ---
def make_prediction(frame_bgr):
    """Make jaundice prediction on a frame"""
    if model is None:
        return None
    
    # Check brightness
    brightness = check_image_brightness(frame_bgr)
    
    # Get brightness threshold from model parameters or use default
    brightness_threshold = model_params.get('brightness_threshold', 35)
    
    # Initialize confidence level
    confidence = 1.0
    
    # Define brightness levels
    very_dark_threshold = brightness_threshold
    low_light_threshold = brightness_threshold * 1.5
    
    # If image is too dark, return early
    if brightness < very_dark_threshold:
        return {
            "status": "too_dark",
            "jaundice_detected": False,
            "confidence": 0.0,
            "probability": 0.0,
            "brightness": float(brightness),
            "reliability": 0.0,
            "message": "Image too dark. Please use better lighting."
        }
    
    # For low light conditions, reduce confidence
    if brightness < low_light_threshold:
        confidence = 0.7
    
    # Preprocess and get model prediction
    img_tensor = preprocess_frame(frame_bgr)
    with torch.no_grad():
        logits = model(img_tensor)
        probability_jaundice = torch.sigmoid(logits).item()
    
    jaundice_detected = probability_jaundice > 0.5
    predicted_class = CLASS_NAMES[1 if jaundice_detected else 0]
    
    result = {
        "status": "success",
        "jaundice_detected": jaundice_detected,
        "predicted_class": predicted_class,
        "confidence": float(probability_jaundice if jaundice_detected else 1 - probability_jaundice),
        "probability": float(probability_jaundice),
        "brightness": float(brightness),
        "reliability": float(confidence),
        "message": f"{predicted_class} detected" if confidence >= 1.0 else f"{predicted_class} detected (low light may affect accuracy)"
    }
    
    return result

# --- Automatic Detection Function ---
async def perform_detection():
    """Perform jaundice detection and return result"""
    global last_detection_result
    
    if model is None:
        logger.error("Model not loaded")
        return None
    
    try:
        # Capture frame from infant camera stream
        logger.info(f"📸 Capturing frame from {INFANT_CAMERA_URL}")
        frame = capture_frame_from_stream(INFANT_CAMERA_URL)
        
        if frame is None:
            logger.error("Failed to capture frame from camera")
            return None
        
        # Make prediction
        result = make_prediction(frame)
        
        if result is None:
            logger.error("Prediction failed")
            return None
        
        result["timestamp"] = datetime.now().isoformat()
        result["detection_type"] = "auto"  # Mark as automatic detection
        
        logger.info(f"✓ Detection result: {result.get('predicted_class', result.get('status'))} "
                   f"(confidence: {result['confidence']:.2%}, brightness: {result['brightness']:.1f})")
        
        # Store last result
        last_detection_result = result
        
        # Publish to ThingsBoard if available - publish every 10 minutes regardless of detection result
        if tb_client and tb_client.enabled:
            # Try to reconnect if not connected
            if not tb_client.connected:
                logger.info("🔄 ThingsBoard not connected, attempting to reconnect...")
                tb_client.connect()
            
            # Publish if connected or result was successful
            if tb_client.connected or result.get('status') == 'success':
                tb_client.publish_jaundice_data(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during automatic detection: {e}")
        return None

async def auto_detection_loop():
    """Background task for automatic detection every 10 minutes"""
    logger.info(f"🔄 Starting automatic detection loop (interval: {AUTO_DETECT_INTERVAL} seconds / 10 minutes)")
    
    while True:
        try:
            # Perform detection
            result = await perform_detection()
            
            if result:
                logger.info(f"✅ Auto-detection completed at {datetime.now().strftime('%H:%M:%S')}")
            else:
                logger.warning(f"⚠️ Auto-detection failed at {datetime.now().strftime('%H:%M:%S')}")
            
            # Wait for next interval
            await asyncio.sleep(AUTO_DETECT_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in auto-detection loop: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry on error

# --- Capture Frame from Stream ---
def capture_frame_from_stream(stream_url):
    """Capture a frame from the MJPG stream"""
    try:
        req = urllib.request.Request(stream_url)
        with urllib.request.urlopen(req, timeout=5) as stream:
            bytes_data = bytes()
            while True:
                bytes_data += stream.read(1024)
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode JPEG
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame
                    break
        return None
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None

# --- API Endpoints ---

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "service": "Jaundice Detection API",
        "version": "2.0.0",
        "status": "running" if model is not None else "error",
        "model_loaded": model is not None,
        "device": DEVICE,
        "auto_detection": {
            "enabled": True,
            "interval_seconds": AUTO_DETECT_INTERVAL,
            "interval_minutes": AUTO_DETECT_INTERVAL / 60,
            "last_detection": last_detection_result.get('timestamp') if last_detection_result else None
        },
        "thingsboard": {
            "enabled": tb_client.enabled if tb_client else False,
            "connected": tb_client.connected if tb_client else False
        },
        "endpoints": {
            "/": "API information",
            "/health": "Service health check",
            "/detect": "Manual jaundice detection from infant camera",
            "/latest": "Get latest detection result (auto or manual)",
            "/model-info": "Get model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "error",
        "model_loaded": model is not None,
        "auto_detection_active": auto_detection_task is not None,
        "thingsboard_connected": tb_client.connected if tb_client else False,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/detect")
async def detect_jaundice():
    """Manual jaundice detection from the infant camera stream"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Capture frame from infant camera stream
        logger.info(f"📸 Manual detection: Capturing frame from {INFANT_CAMERA_URL}")
        frame = capture_frame_from_stream(INFANT_CAMERA_URL)
        
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame from camera")
        
        # Make prediction
        result = make_prediction(frame)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        result["timestamp"] = datetime.now().isoformat()
        result["detection_type"] = "manual"  # Mark as manual detection
        
        logger.info(f"✓ Manual detection: {result.get('predicted_class', result.get('status'))} "
                   f"(confidence: {result['confidence']:.2%})")
        
        # Store as last result
        global last_detection_result
        last_detection_result = result
        
        # Publish to ThingsBoard if available - publish every detection regardless of result
        if tb_client and tb_client.enabled:
            # Try to reconnect if not connected
            if not tb_client.connected:
                logger.info("🔄 ThingsBoard not connected, attempting to reconnect...")
                tb_client.connect()
            
            # Publish if connected or result was successful
            if tb_client.connected or result.get('status') == 'success':
                tb_client.publish_jaundice_data(result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error during manual detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest")
async def get_latest_detection():
    """Get the latest detection result (from auto or manual detection)"""
    if last_detection_result is None:
        return {
            "status": "no_detection",
            "message": "No detection has been performed yet",
            "timestamp": datetime.now().isoformat()
        }
    
    return JSONResponse(content=last_detection_result)

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_path": MODEL_PATH,
        "model_loaded": model is not None,
        "device": DEVICE,
        "image_size": IMG_SIZE,
        "class_names": CLASS_NAMES,
        "model_parameters": model_params,
        "camera_source": INFANT_CAMERA_URL
    }

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Load model and start auto-detection on startup"""
    global model, model_params, tb_client, auto_detection_task
    
    logger.info("🚀 Starting Jaundice Detection Server...")
    logger.info(f"📂 Loading model from {MODEL_PATH}")
    
    # Load ML model
    model, model_params = load_trained_model(MODEL_PATH)
    
    if model is None:
        logger.error("❌ Failed to load model")
    else:
        logger.info("✅ Model loaded successfully")
        logger.info(f"📊 Model parameters: {model_params}")
    
    logger.info(f"📹 Using camera stream: {INFANT_CAMERA_URL}")
    logger.info(f"🌐 Server ready on http://localhost:8887")
    
    # Initialize ThingsBoard client
    if ACCESS_TOKEN:
        logger.info("🔗 Initializing ThingsBoard connection...")
        tb_client = ThingsBoardClient()
        
        # Connect in background thread to not block startup
        def connect_tb():
            if tb_client.connect():
                logger.info("✅ ThingsBoard connected successfully")
            else:
                logger.warning("⚠️ ThingsBoard connection failed, will retry during publishing")
        
        threading.Thread(target=connect_tb, daemon=True).start()
    else:
        logger.warning("⚠️ ThingsBoard not configured, running without cloud integration")
    
    # Start automatic detection loop
    logger.info(f"🔄 Starting automatic detection (every {AUTO_DETECT_INTERVAL / 60:.0f} minutes)")
    auto_detection_task = asyncio.create_task(auto_detection_loop())
    logger.info("✅ Automatic detection started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global auto_detection_task, tb_client
    
    logger.info("🛑 Shutting down Jaundice Detection Server...")
    
    # Cancel auto-detection task
    if auto_detection_task:
        auto_detection_task.cancel()
        try:
            await auto_detection_task
        except asyncio.CancelledError:
            pass
        logger.info("✓ Auto-detection stopped")
    
    # Disconnect from ThingsBoard
    if tb_client:
        tb_client.disconnect()
        logger.info("✓ ThingsBoard disconnected")
    
    logger.info("✅ Shutdown complete")

# --- Main ---
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8887,
        log_level="info"
    )
