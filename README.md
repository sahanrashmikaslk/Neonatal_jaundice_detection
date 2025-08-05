# Advanced Neonatal Jaundice Detection System

## Project Overview

This project implements a state-of-the-art machine learning system for detecting neonatal jaundice from images of infant eyes or skin. The system features a **lighting-robust** deep learning model combined with an intuitive Streamlit web application, providing reliable detection even under challenging lighting conditions.

### Key Innovation: Lighting-Robust Detection

Unlike traditional models that may produce false positives in poor lighting, our system includes:

- **Automatic brightness detection** to identify low-light conditions
- **Adaptive confidence scoring** that reduces reliability when lighting is insufficient
- **Enhanced image processing** using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **"Too Dark" classification** to prevent false diagnoses in inadequate lighting

## Model Architecture & Performance

### Core Model Details

- **Architecture:** MobileNetV3-Small (fine-tuned for medical imaging)
- **Framework:** PyTorch with ONNX compatibility for deployment
- **Training Data:** [Kaggle Jaundice Image Data](https://www.kaggle.com/datasets/aiolapo/jaundice-image-data)
  - ~200 Jaundice cases
  - ~560 Normal cases
- **Input:** 224×224 RGB images
- **Output:** Binary classification (Normal/Jaundice) with confidence scores

### Enhanced Features

- **Dual Model System:** Base model + Lighting-robust wrapper
- **Brightness Threshold:** Configurable detection of dark images (default: 35-70 brightness units)
- **Confidence Levels:** Full confidence (1.0) for good lighting, reduced (0.7) for low light
- **ONNX Export:** Ready for Raspberry Pi and edge device deployment

## Training Process Deep Dive

The model training pipeline (implemented in `jaundice-detection.ipynb`) includes:

### 1. **Data Acquisition & EDA**

```python
# Automated Kaggle dataset download
!kaggle datasets download -d aiolapo/jaundice-image-data
```

- Comprehensive exploratory data analysis
- Class distribution visualization
- Sample image inspection

### 2. **Custom Dataset Implementation**

```python
class EyeJaundiceSet(Dataset):
    # Handles RGB conversion, resizing, augmentation
    # 85/15 train/validation split with reproducible seeding
```

### 3. **Transfer Learning Setup**

- Pre-trained MobileNetV3-Small with ImageNet weights
- Custom binary classification head
- BCEWithLogitsLoss for stable training
- AdamW optimizer with ReduceLROnPlateau scheduling

### 4. **Advanced Training Loop**

- 10-epoch training with comprehensive metrics
- Windows-compatible DataLoader (num_workers=0)
- Real-time accuracy, sensitivity, and specificity tracking
- Progress bars with tqdm integration

### 5. **Lighting-Robust Enhancement**

```python
class LightingRobustJaundiceModel:
    def __init__(self, base_model, brightness_threshold=70):
        # Wrapper that adds lighting awareness to base model

    def is_dark_image(self, image):
        # Calculates average brightness in grayscale

    def enhance_image(self, image):
        # CLAHE enhancement for low-light conditions
```

### 6. **Model Evaluation & Validation**

- Comparative analysis: Base model vs. Lighting-robust model
- False positive rate reduction measurement
- Dark image detection statistics
- Comprehensive metric reporting (accuracy, sensitivity, specificity)

### 7. **Export & Deployment**

- **PyTorch Format:** `jaundice_mobilenetv3_robust.pt` (includes all robust parameters)
- **ONNX Format:** `jaundice_mobilenetv3_robust.onnx` (cross-platform compatibility)
- Embedded model parameters (brightness thresholds, normalization constants)

## Advanced Streamlit Application

### Core Features

- **Multi-Input Support:** File upload, webcam snapshot, live feed
- **Real-time Analysis:** Frame-by-frame processing for live detection
- **Intelligent Feedback:** Color-coded results with confidence indicators
- **Model Parameter Display:** Transparency in model configuration

### Lighting-Aware Interface

```python
def display_prediction_text(predicted_class, probability, brightness, confidence):
    if predicted_class == "Too Dark":
        # Orange warning for insufficient lighting
    elif predicted_class == "Jaundice":
        # Red alert with reliability indicator
    else:  # Normal
        # Green confirmation with confidence metrics
```

### Live Feed Enhancements

- **Background Processing:** Non-blocking live detection
- **Visual Overlays:** Real-time brightness and reliability indicators
- **Smooth State Management:** Proper camera resource handling
- **Performance Optimized:** Efficient frame processing pipeline

## Project Structure

```
Neonatal_jaundice_detection/
├── jaundice-detection.ipynb          # Complete training pipeline
├── app.py                           # Enhanced Streamlit application
├── requirements.txt                 # Python package dependencies
├── jaundice_mobilenetv3.pt          # Base model weights
├── jaundice_mobilenetv3_robust.pt   # Lighting-robust model
├── jaundice_mobilenetv3.onnx         # Base ONNX export
├── jaundice_mobilenetv3_robust.onnx  # Robust ONNX export
├── data/                            # Training dataset
│   ├── jaundice-image-data.zip
│   ├── jaundice/                       # Positive cases
│   └── normal/                         # Negative cases
├── jaundice-env/                    # Virtual environment
├── ScreenShots/                     # Application demos
├── prev_script/                     # Development history
└── README.md                        # This documentation
```

## Quick Start Guide

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, CPU supported)
- Webcam access for live features

### Installation

```bash
# Clone repository
git clone https://github.com/sahanrashmikaslk/Neonatal_jaundice_detection.git
cd Neonatal_jaundice_detection

# Create virtual environment
python -m venv jaundice-env
# Windows
.\jaundice-env\Scripts\activate
# Linux/macOS
source jaundice-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

### For Training/Development

```bash
# Setup Kaggle API (place kaggle.json in .kaggle folder)
# Open jaundice-detection.ipynb in Jupyter/VS Code
# Run all cells to reproduce training process
```

## Application Interface

### Upload Analysis

- Support for JPG, PNG, JPEG formats
- Instant brightness assessment
- Detailed confidence reporting

![Upload an Image](./ScreenShots/UploadAnImage.png)

### Webcam Integration

- Real-time camera access
- Snapshot analysis with lighting validation
- User-friendly capture interface

![Webcam Snapshot](./ScreenShots/UseWebcamSnapshot.png)

### Live Feed Detection

- Continuous frame-by-frame analysis
- Real-time brightness monitoring
- Visual reliability indicators
- Smooth start/stop controls

![Live Feed Detection](./ScreenShots/LiveFeedDetection.png)

## Usage Best Practices

### For Optimal Results

1. **Lighting:** Ensure adequate, even illumination
2. **Focus:** Target the sclera (white part) of the eye
3. **Stability:** Minimize movement during capture
4. **Distance:** Maintain appropriate camera distance

### Understanding Outputs

- **"Normal":** No jaundice detected (Green indicator)
- **"Jaundice":** Potential jaundice detected (Red indicator)
- **"Too Dark":** Insufficient lighting (Orange warning)
- **Reliability Score:** Confidence level based on lighting conditions

## Technical Innovations

### Brightness-Aware Prediction

```python
def make_prediction_on_frame(model, frame, model_params):
    brightness = check_image_brightness(frame)
    if brightness < brightness_threshold:
        return "Too Dark", 0, brightness, 0.0
    # Continue with normal prediction...
```

### Adaptive Confidence Scoring

- **Full Confidence (1.0):** Good lighting conditions
- **Reduced Confidence (0.7):** Low light but analyzable
- **No Confidence (0.0):** Too dark for reliable analysis

### CLAHE Enhancement

- Contrast Limited Adaptive Histogram Equalization
- Improves visibility in challenging lighting
- Maintains color accuracy for medical assessment

## Deployment Options

### Local Development

- Streamlit application for research and testing
- Full model transparency and parameter access

### Production Deployment

- ONNX models for cross-platform compatibility
- Raspberry Pi ready for edge deployment
- Lightweight inference pipeline

### Clinical Integration

- API-ready model structure
- Standardized input/output formats
- Comprehensive logging and audit trails

## Performance Metrics

The lighting-robust model demonstrates:

- **Improved Specificity:** Reduced false positive rates in low light
- **Maintained Sensitivity:** Preserved detection capability
- **Enhanced Reliability:** Transparent confidence scoring
- **Real-world Robustness:** Handles varying lighting conditions

## Contributing

This project welcomes contributions in:

- Model architecture improvements
- Additional data augmentation techniques
- Enhanced user interface features
- Clinical validation studies
- Performance optimization

## Medical Disclaimer

This system is designed for **research and educational purposes**. It should not replace professional medical diagnosis. Always consult qualified healthcare providers for medical decisions regarding neonatal jaundice.

## Contact & Support

For questions, issues, or collaboration opportunities, please reach out through the project repository or contact the development team.
