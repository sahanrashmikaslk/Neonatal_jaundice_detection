
# jaundice_detector.py - Standalone detector with lighting robustness
import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import os

class LightingRobustJaundiceModel:
    def __init__(self, model_path='jaundice_mobilenetv3.pt', device='cpu', brightness_threshold=70):
        # Set up the device
        self.device = device
        self.brightness_threshold = brightness_threshold
        
        # Load model
        self.model = mobilenet_v3_small()
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 1)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Image constants
        self.img_size = 224
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
    
    def enhance_image(self, image):
        # Enhance dark image using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def is_dark_image(self, image):
        # Check if image is too dark
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < self.brightness_threshold
    
    def preprocess(self, image):
        # Resize
        img = cv2.resize(image, (self.img_size, self.img_size))
        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0
        # Standardize
        img = (img - np.array(self.mean)) / np.array(self.std)
        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return img.to(self.device)
    
    def predict(self, image):
        # Check lighting
        is_dark = self.is_dark_image(image)
        
        # For dark images, enhance
        if is_dark:
            image = self.enhance_image(image)
        
        # Preprocess and predict
        tensor = self.preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits).item()
        
        # Calculate confidence based on lighting
        confidence = 1.0 if not is_dark else 0.7
        
        return {
            'jaundice_probability': probability,
            'confidence': confidence,
            'is_dark_image': is_dark
        }

# Demo usage
if __name__ == "__main__":
    import sys
    
    # Initialize detector
    detector = LightingRobustJaundiceModel()
    
    # Check if image file provided
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        # Read image
        img = cv2.imread(sys.argv[1])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        result = detector.predict(img_rgb)
        
        # Display result
        print(f"Jaundice probability: {result['jaundice_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Dark image: {result['is_dark_image']}")
        
        # Draw on image
        status = "JAUNDICE" if result['jaundice_probability'] > 0.5 else "NORMAL"
        color = (0, 0, 255) if result['jaundice_probability'] > 0.5 else (0, 255, 0)
        
        img_display = img.copy()
        cv2.putText(img_display, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img_display, f"Prob: {result['jaundice_probability']:.2f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if result['is_dark_image']:
            cv2.putText(img_display, "LOW LIGHT WARNING", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        # Show image
        cv2.imshow("Jaundice Detector", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Please provide an image path")
        print("Usage: python jaundice_detector.py <image_path>")
