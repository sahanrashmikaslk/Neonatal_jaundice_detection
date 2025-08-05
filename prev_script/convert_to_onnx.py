"""
Convert PyTorch model to ONNX format for Raspberry Pi deployment
"""

import torch
import torchvision.models as models
import torch.nn as nn
import os

# Configuration - must match your trained model
MODEL_PATH = "jaundice_mobilenetv3.pt"
ONNX_MODEL_PATH = "jaundice_mobilenetv3.onnx"
IMG_SIZE = 224

# Define the model architecture (must match your training architecture)
def get_model_architecture():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model

def convert_model_to_onnx():
    # Load model
    print(f"Loading PyTorch model from {MODEL_PATH}")
    model = get_model_architecture()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    
    # Export to ONNX
    print(f"Converting model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    
    # Verify file size
    onnx_size = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)  # Convert to MB
    print(f"Successfully exported model to {ONNX_MODEL_PATH} ({onnx_size:.2f} MB)")

if __name__ == "__main__":
    convert_model_to_onnx()
