"""
RETINOVA: Explainable AI Framework for Early Diabetic Retinopathy Detection
Streamlit application for diabetic retinopathy screening from retinal fundus images.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import torch
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "retinova_efficientnetv2.pth"
CLASS_NAMES = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
IMG_SIZE = 224

st.set_page_config(page_title="RETINOVA", layout="wide")


@st.cache_resource
def load_model():
    """Load the trained EfficientNetV2 classifier and select inference device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(img_array, img_size=IMG_SIZE):
    """
    Apply CLAHE-based contrast enhancement and resize the fundus image.
    CLAHE is applied on the L-channel in LAB color space to improve
    visibility of retinal lesions without over-amplifying noise.
    """
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB) if img_array.shape[-1] == 3 else img_array
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    img_resized = cv2.resize(img_clahe, (img_size, img_size))
    return img_resized


def get_transform():
    """Normalization transform matching the training pipeline."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_recommendation(pred_class, confidence):
    """Map predicted grade and confidence to a referral recommendation."""
    if confidence < 70:
        return "Borderline — Refer for Manual Review"
    elif pred_class == 0:
        return "Clear — Routine Follow-up"
    elif pred_class in [1, 2]:
        return "Monitor — Periodic Screening"
    else:
        return "Refer — Immediate Ophthalmologist Review"


# ---------------- UI ----------------
st.title("RETINOVA — AI-Assisted Diabetic Retinopathy Screening")
st.caption("Explainable AI framework for early diabetic retinopathy detection")

st.sidebar.header("Patient Details")
patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)

uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_img = np.array(Image.open(uploaded_file).convert("RGB"))
    processed_img = preprocess_image(original_img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_img, use_container_width=True)
    with col2:
        st.subheader("Preprocessed (CLAHE Enhanced)")
        st.image(processed_img, use_container_width=True)

    if st.button("Analyze Image"):
        try:
            model, device = load_model()

            transform = get_transform()
            tensor_img = transform(image=processed_img)['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor_img)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item() * 100

            st.markdown("---")
            st.subheader("Screening Result")

            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Grade", CLASS_NAMES[pred_class])
            r2.metric("Confidence", f"{confidence:.1f}%")

            recommendation = get_recommendation(pred_class, confidence)
            r3.metric("Recommendation", recommendation.split(" — ")[0])
            st.markdown(f"### {recommendation}")

            st.subheader("Class-wise Probability")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(CLASS_NAMES, probs.cpu().numpy())
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

            st.subheader("Explainability — Grad-CAM Heatmap")
            target_layer = [model.conv_head]
            cam = GradCAM(model=model, target_layers=target_layer)
            cam_targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(input_tensor=tensor_img, targets=cam_targets)[0]
            visualization = show_cam_on_image(processed_img / 255.0, grayscale_cam, use_rgb=True)
            st.image(visualization, caption="Regions influencing the model's prediction", use_container_width=True)

        except FileNotFoundError:
            st.error(
                f"Model file '{MODEL_PATH}' not found. Place the trained '.pth' file "
                f"in the same directory as this script."
            )
else:
    st.info("Upload a fundus image to begin analysis.")