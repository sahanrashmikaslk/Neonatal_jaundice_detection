import torch
import torchvision.models as models
import torch.nn as nn

def get_model_architecture():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model

def inspect_model(model_path):
    print(f"\nInspecting model: {model_path}")
    try:
        # Load the model architecture
        model = get_model_architecture()
        
        # Try loading the state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            print("Model contains a 'state_dict' key - extracting the actual state dict")
            state_dict = state_dict['state_dict']
        
        # Print state dict keys
        print(f"State dict keys: {state_dict.keys()}")
        
        # Try loading the state dict into the model
        try:
            model.load_state_dict(state_dict)
            print("Successfully loaded state dict into model")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            
        return True
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return False

# Inspect both models
inspect_model("jaundice_mobilenetv3.pt")
inspect_model("jaundice_mobilenetv3_robust.pt")
