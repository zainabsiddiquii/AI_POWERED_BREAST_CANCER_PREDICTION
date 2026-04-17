import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime

# ===============================
# 🎀 Page Configuration
# ===============================
st.set_page_config(
    page_title="AI Disease Prediction - Breast Cancer Detection",
    page_icon="🩺",
    layout="wide",
)

# ===============================
# 🎀 Session State for History
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# 🎀 Custom CSS
# ===============================
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color: #ff4b4b;
            font-size: 42px;
            font-weight: bold;
        }
        .sub-title {
            text-align: center;
            color: gray;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 🎀 Load Model
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()

MODEL_IMG_SIZE = model.input_shape[1]

# ===============================
# 🎀 Enhanced Sidebar
# ===============================
st.sidebar.header("⚙️ Model Settings")

if st.sidebar.button("🧹 Clear History"):
    st.session_state.history = []

display_mode = st.sidebar.selectbox(
    "Prediction Display Mode",
    ["Show Label Only", "Show Label + Confidence"]
)

heatmap_strength = st.sidebar.slider(
    "🔥 Grad-CAM Intensity",
    0.1, 1.0, 0.4
)

st.sidebar.markdown("---")
st.sidebar.subheader("📘 Model Information")
st.sidebar.info(f"""
Model Type: CNN + XAI  
Input Size: {MODEL_IMG_SIZE}x{MODEL_IMG_SIZE}  
Classes: benign, malignant, normal  
Explainability: Grad-CAM Enabled
""")

# ===============================
# 🎯 Grad-CAM Function
# ===============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):

    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ===============================
# 🎀 Hero Section
# ===============================
st.markdown("""
<div class='main-title'>🩺 AI Breast Cancer Detection System</div>
<div class='sub-title'>CNN Integrated with Explainable AI (XAI)</div>
""", unsafe_allow_html=True)

st.write("""
This intelligent system predicts breast cancer from histopathology images
and provides explainable visual insights using Grad-CAM.
""")

# ===============================
# 🎀 Upload Section
# ===============================
uploaded_file = st.file_uploader(
    "📸 Upload a histopathology image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):

        # ===============================
        # Preprocessing
        # ===============================
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE))

        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ===============================
        # Prediction
        # ===============================
        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        classes = ['benign', 'malignant', 'normal']
        result = classes[predicted_class]

        confidence = round(np.max(prediction) * 100, 2)

        # ===============================
        # Display Result
        # ===============================
        if display_mode == "Show Label Only":
            st.success(f"Prediction: {result}")
            history_text = result
        else:
            st.success(f"Prediction: {result}")
            st.write(f"Confidence: {confidence}%")
            history_text = f"{result} ({confidence}%)"

        st.session_state.history.append(history_text)

        # ===============================
        # 🚨 Risk Indicator
        # ===============================
        if result == "malignant":
            st.error("🔴 High Risk Detected")
        elif result == "benign":
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

        # ===============================
        # 📊 Probability Chart
        # ===============================
        st.subheader("📊 Prediction Probability")

        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability": prediction[0] * 100
        })

        st.bar_chart(prob_df.set_index("Class"))

        # ===============================
        # 🩺 Medical Recommendation
        # ===============================
        with st.expander("🩺 Medical Recommendation"):
            if result == "malignant":
                st.write("Immediate consultation with an oncologist is recommended.")
            elif result == "benign":
                st.write("Regular follow-up and periodic monitoring is advised.")
            else:
                st.write("No abnormality detected. Continue regular screenings.")

        # ===============================
        # 🔥 Grad-CAM Visualization
        # ===============================
        st.subheader("🔥 Grad-CAM Visualization")

        heatmap = make_gradcam_heatmap(img_array, model)

        heatmap = cv2.resize(heatmap, (MODEL_IMG_SIZE, MODEL_IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)

        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = np.uint8(jet_heatmap * 255)

        superimposed_img = jet_heatmap * heatmap_strength + np.array(image)
        superimposed_img = np.uint8(superimposed_img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            st.image(superimposed_img, caption="Grad-CAM Highlight", use_column_width=True)

        # ===============================
        # 🕒 Timestamp
        # ===============================
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"Prediction Time: {current_time}")

        # ===============================
        # 📥 Download Report
        # ===============================
        report_text = f"""
Breast Cancer Detection Report

Prediction: {result}
Confidence: {confidence}%
Time: {current_time}
"""

        st.download_button(
            "📥 Download Prediction Report",
            report_text,
            file_name="prediction_report.txt"
        )

# ===============================
# 🎀 Prediction History
# ===============================
st.subheader("📜 Previous Predictions")

if st.session_state.history:
    history_df = pd.DataFrame(
        {"Predictions": st.session_state.history}
    )
    st.table(history_df)
else:
    st.info("No predictions yet.")

# ===============================
# 🎀 Footer
# ===============================
st.markdown("""
---
Made by **Zainab Siddiqui, Sehba Fatima and Noor Fatima**  
Final Year Project — AI Disease Prediction  

*Making prediction easier with AI-powered medical imaging.*
""")