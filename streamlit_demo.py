# demo.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import numpy as np
from ultralytics import YOLO
import torchvision.models as models

# -----------------------
# CONFIG
# -----------------------
TARGET_RES = (288, 288)
YOLO_MODEL_PATH = "yolo11n.pt"  # Update if needed
CLASSIFIER_PATH = "multi_item_model_shard_38.pth"  # Update if needed, change this to match model name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_ROOT = "C:/Users/ekhou/Downloads/AI-Powered Nutrition Assitant/food-101/food-101/food-101/images"


# -----------------------
# Load Class Names from Food-101 Folder
# -----------------------
@st.cache_resource
def get_class_names(image_root=IMAGE_ROOT):
    if not os.path.exists(image_root):
        return []
    return sorted([
        d for d in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, d)) and not d.startswith(".")
    ])

CLASS_NAMES = get_class_names()

# -----------------------
# Load YOLO Model
# -----------------------
@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)

# -----------------------
# Load Classifier Model
# -----------------------
# Load class names BEFORE creating model


@st.cache_resource
def load_classifier():
    model = models.resnet50()

    # Load checkpoint
    state_dict = torch.load(CLASSIFIER_PATH, map_location="cpu")

    # Remove classifier weights if shapes mismatch
    if "fc.weight" in state_dict and state_dict["fc.weight"].shape[0] != len(CLASS_NAMES):
        print(f"⚠️ Removing fc.weight due to size mismatch: {state_dict['fc.weight'].shape} vs {len(CLASS_NAMES)}")
        del state_dict["fc.weight"]
    if "fc.bias" in state_dict and state_dict["fc.bias"].shape[0] != len(CLASS_NAMES):
        print(f"⚠️ Removing fc.bias due to size mismatch: {state_dict['fc.bias'].shape} vs {len(CLASS_NAMES)}")
        del state_dict["fc.bias"]

    # Load everything else
    model.load_state_dict(state_dict, strict=False)

    # Rebuild fc with correct output size
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    model.to(DEVICE)
    model.eval()
    return model



# -----------------------
# Transform
# -----------------------
crop_transform = transforms.Compose([
    transforms.Resize(TARGET_RES),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------
# Inference Pipeline
# -----------------------
def run_yolo_and_classify(img, yolo_model, classifier_model):
    results = yolo_model.predict(img, conf=0.25, device=DEVICE, verbose=False)[0]
    draw = ImageDraw.Draw(img)
    predictions = []

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        cropped = img.crop((x1, y1, x2, y2))
        transformed = crop_transform(cropped).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = classifier_model(transformed)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = CLASS_NAMES[pred_idx]
            predictions.append((pred_class, (x1, y1, x2, y2)))

            # Draw box + label
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 3, y1 + 3), pred_class.replace("_", " ").title(), fill="white")

    return img, predictions

# -----------------------
# Streamlit UI
# -----------------------
st.title("🍽️ AI-Powered Food Detector & Classifier")

uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    orig = Image.open(uploaded_file).convert("RGB")
    st.image(orig, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running detection and classification..."):
        yolo_model = load_yolo_model()
        classifier_model = load_classifier()
        result_img, preds = run_yolo_and_classify(orig.copy(), yolo_model, classifier_model)

    st.subheader("Results")
    if preds:
        st.image(result_img, caption="Detected Items", use_container_width=True)
        for label, box in preds:
            st.write(f"**{label.replace('_', ' ').title()}** at {box}")
    else:
        st.warning("No food items detected.")
