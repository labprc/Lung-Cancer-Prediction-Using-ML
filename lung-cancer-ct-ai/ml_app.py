import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import joblib

st.set_page_config(page_title="Lung Cancer CT - Traditional ML", layout="wide")

MODEL_PATH = "results/traditional_ml_model.pkl"
MALIGNANT_CLASSES = ['adenocarcinoma', 'large cell carcinoma', 'squamous cell carcinoma']
CONFIDENCE_THRESHOLD = 0.60

@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model_data = load_ml_model()

st.title("🫁 Lung Cancer CT Classification (Traditional ML)")
st.markdown("**Powered by Traditional Machine Learning (Random Forest) — No Deep Learning/AI.**")
st.markdown("**Educational / research prototype only. Not a medical diagnostic tool.**")
st.markdown(f"*Malignant classes: {', '.join(MALIGNANT_CLASSES)}*")

uploaded_file = st.file_uploader("Upload CT Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model_data is not None:
    clf = model_data['model']
    classes = model_data['classes']
    img_size = model_data['img_size']

    # Process Image exactly as we did in training
    image = Image.open(uploaded_file).convert('RGB')
    
    # Convert to OpenCV format (BGR then Grayscale)
    img_cv = np.array(image) 
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Resize and Flatten
    img_resized = cv2.resize(img_gray, (img_size, img_size))
    img_features = img_resized.flatten().reshape(1, -1)

    # Predict Probabilities
    preds = clf.predict_proba(img_features)[0]
    pred_idx = np.argmax(preds)
    pred_class = classes[pred_idx]
    confidence = preds[pred_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded CT Scan", use_container_width=True)
        
    with col2:
        st.subheader("Machine Learning Prediction:")
        st.write(f"### {pred_class.upper()}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("STATUS: UNCERTAIN\n\nModel confidence is low. This result should not be treated as a reliable classification.")
        else:
            st.success("STATUS: HIGH CONFIDENCE")

        if pred_class in MALIGNANT_CLASSES:
            st.error("⚠️ Malignant classification detected. This is a research prototype. Please consult a qualified radiologist.")

        st.write("---")
        st.write("**All Probabilities (Random Forest Voting):**")
        for i, cls in enumerate(classes):
            st.write(f"- {cls}: {preds[i]*100:.2f}%")

elif model_data is None:
    st.error("Traditional ML Model not found! Please run `python ml_pipeline.py` first.")
