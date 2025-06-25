import torch
import os
from torchvision.models import resnet50, ResNet50_Weights

try:
    print("⏬ Downloading ResNet-50 weights...")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "resnet50-weights.pth")
    torch.save(model.state_dict(), downloads_path)

    print(f"✅ ResNet-50 weights saved to: {downloads_path}")

except Exception as e:
    print(f"❌ Error: {e}")
