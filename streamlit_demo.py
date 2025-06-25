# demo.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

from evaluate_model import (
    plot_interactive_confusion_matrix,
    evaluate_and_save_preds  # <- wrap this in your evaluate_model.py
)

# -------------------------
# CONFIG
# -------------------------
IMAGE_ROOT = "C:/Users/ekhou/Downloads/AI-Powered Nutrition Assitant/food-101/food-101/food-101/images"
MODEL_PATH = "C:/Users/ekhou/Downloads/AI-Powered Nutrition Assitant/fast_cnn_with_fp.pth"
PRED_PATH = "C:/Users/ekhou/Downloads/AI-Powered Nutrition Assitant/eval_preds.npy"
LABEL_PATH = "C:/Users/ekhou/Downloads/AI-Powered Nutrition Assitant/eval_labels.npy"
TARGET_RES = (288, 288)

CLASS_NAMES = sorted([
    d for d in os.listdir(IMAGE_ROOT)
    if os.path.isdir(os.path.join(IMAGE_ROOT, d)) and not d.startswith(".")
])

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize(TARGET_RES),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("Food Image Classifier")
st.write("Upload a food image to get a prediction or view evaluation results.")

col1, col2, col3 = st.columns(3)

# ----------- Upload and Classify Image -----------
with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner("Classifying..."):
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Uploaded Image", use_container_width=True)

                input_tensor = transform(img).unsqueeze(0).to("cpu")

                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    pred_class = CLASS_NAMES[pred_idx]

                st.success(f"Predicted: **{pred_class.replace('_', ' ').title()}**")

            except Exception as e:
                st.error(f"Error loading or classifying image: {e}")

# ----------- Confusion Matrix -----------
with col2:
    if st.button("Show Confusion Matrix"):
        try:
            if not (os.path.exists(PRED_PATH) and os.path.exists(LABEL_PATH)):
                with st.spinner("Running evaluation..."):
                    evaluate_and_save_preds()

            all_preds = np.load(PRED_PATH)
            all_labels = np.load(LABEL_PATH)

            fig = plot_interactive_confusion_matrix(
                y_true=all_labels,
                y_pred=all_preds,
                class_names=CLASS_NAMES,
                normalize=False
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load or render confusion matrix: {e}")

# ----------- Loss & Accuracy Graph -----------
with col3:
    if st.button("Show Loss and Validation Accuracy Graph"):
        try:
            loss_data = np.load("train_losses.npy")
            val_acc_data = np.load("val_accuracies.npy")

            fig, ax1 = plt.subplots(figsize=(8, 4))

            ax1.plot(loss_data, color='red')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss', color='red')
            ax1.tick_params(axis='y', labelcolor='red')

            ax2 = ax1.twinx()
            ax2.plot(val_acc_data, color='blue')
            ax2.set_ylabel('Validation Accuracy', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

            plt.title("Training Loss & Validation Accuracy")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Could not load or render training graph: {e}")


