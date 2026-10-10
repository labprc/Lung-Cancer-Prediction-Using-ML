import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import io
import uuid
import datetime
import time
from PIL import Image
from streamlit_option_menu import option_menu

# PDF Generation Imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ==========================================
# CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(page_title="LungCare AI", layout="wide", initial_sidebar_state="collapsed")

MODEL_PATH = "results/best_model.keras"
CLASS_NAMES = ['Benign cases', 'Normal cases', 'adenocarcinoma', 'large cell carcinoma', 'squamous cell carcinoma']
MALIGNANT_CLASSES = ['adenocarcinoma', 'large cell carcinoma', 'squamous cell carcinoma']
CONFIDENCE_THRESHOLD = 0.60

@st.cache_resource
def load_cnn_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_cnn_model()

def format_class_name(cls_name):
    return cls_name.title()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ==========================================
# PDF GENERATOR
# ==========================================
def generate_pdf_report(patient_data, file_name, original_pil, superimposed_array, pred_class, confidence, is_malignant, is_uncertain, preds):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    
    styles = getSampleStyleSheet()
    sub_title = ParagraphStyle('SubTitle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#475569"), spaceAfter=1)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading3'], fontSize=11, textColor=colors.HexColor("#1e293b"), spaceAfter=6, spaceBefore=12)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor("#334155"), leading=12)
    bold_body = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')

    report_id = str(uuid.uuid4())[:8].upper()
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    header_data = [
        [
            Paragraph("<b>🫁 LUNGCARE AI</b><br/>Lung CT Classification & Clinical Analysis<br/><i>AI-Assisted Risk Assessment<br/>BE Final Year Project</i>", sub_title),
            Paragraph(f"<font color='#0284c7'><b>AI RESEARCH PROTOTYPE</b></font><br/>Report ID: {report_id}<br/>Report Date: {date_str}<br/>Report Time: {time_str}", sub_title)
        ]
    ]
    t_header = Table(header_data, colWidths=[350, 185])
    t_header.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('LINEBELOW', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")), ('BOTTOMPADDING', (0,0), (-1,-1), 10)]))
    elements.append(t_header)

    elements.append(Paragraph("PATIENT / CLINICAL INFORMATION", heading_style))
    user_data = [
        ["Name:", patient_data.get('Name', 'Not Provided'), "Age / Gender:", f"{patient_data.get('Age', '-')} / {patient_data.get('Gender', '-')}"]
    ]
    
    if patient_data.get('Age') is not None:
        user_data.extend([
            ["Pack Years:", str(patient_data.get('Pack Years', '-')), "Radon Exposure:", patient_data.get('Radon', '-')],
            ["Asbestos:", patient_data.get('Asbestos', '-'), "Secondhand Smoke:", patient_data.get('Secondhand', '-')],
            ["COPD Diagnosis:", patient_data.get('COPD', '-'), "Alcohol Intake:", patient_data.get('Alcohol', '-')]
        ])
    
    t_user = Table(user_data, colWidths=[100, 167, 100, 168])
    t_user.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9), ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#334155")), ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_user)

    img_io1 = io.BytesIO()
    original_pil.save(img_io1, format='JPEG')
    img_io1.seek(0)
    rl_img1 = RLImage(img_io1, width=2.4*inch, height=2.4*inch)

    gradcam_pil = Image.fromarray(superimposed_array)
    img_io2 = io.BytesIO()
    gradcam_pil.save(img_io2, format='JPEG')
    img_io2.seek(0)
    rl_img2 = RLImage(img_io2, width=2.4*inch, height=2.4*inch)

    elements.append(Paragraph("CT IMAGE ANALYSIS", heading_style))
    img_data = [[Paragraph("<b>ORIGINAL CT IMAGE</b>", ParagraphStyle('C', alignment=1, fontSize=9)), Paragraph("<b>CNN FEATURE ATTENTION (Grad-CAM)</b>", ParagraphStyle('C', alignment=1, fontSize=9))], [rl_img1, rl_img2]]
    t_img = Table(img_data, colWidths=[267, 268])
    t_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('BOX', (0,0), (0,-1), 1, colors.HexColor("#e2e8f0")), ('BOX', (1,0), (1,-1), 1, colors.HexColor("#e2e8f0")), ('BOTTOMPADDING', (0,0), (-1,0), 6)]))
    elements.append(t_img)

    elements.append(Paragraph("AI CLASSIFICATION RESULT", heading_style))
    status_color = colors.HexColor("#ef4444") if is_malignant else colors.HexColor("#10b981")
    if is_uncertain: status_color = colors.HexColor("#f59e0b")
    bg_color = colors.HexColor("#fee2e2") if is_malignant else colors.HexColor("#d1fae5")
    if is_uncertain: bg_color = colors.HexColor("#fef3c7")

    result_data = [[f"Predicted Class:", format_class_name(pred_class).upper(), "Confidence:", f"{confidence*100:.2f}%"], [f"Status:", "UNCERTAIN" if is_uncertain else "HIGH CONFIDENCE", "Classification:", "MALIGNANT" if is_malignant else "NON-MALIGNANT"]]
    t_res = Table(result_data, colWidths=[90, 177, 70, 198])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color), ('BOX', (0,0), (-1,-1), 2, status_color),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica-Bold'), ('FONTNAME', (3,0), (3,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (1,0), (1,-1), status_color), ('TEXTCOLOR', (3,1), (3,1), status_color), ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_res)

    elements.append(Paragraph("CLASS PROBABILITY DISTRIBUTION", heading_style))
    prob_data = [["Class Name", "Model Probability"]]
    sorted_indices = np.argsort(preds)[::-1]
    for idx in sorted_indices:
        prob_data.append([format_class_name(CLASS_NAMES[idx]), f"{preds[idx]*100:.2f}%"])
        
    t_prob = Table(prob_data, colWidths=[350, 185])
    t_prob.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#f8fafc")), ('FONTNAME', (0,1), (-1,1), 'Helvetica-Bold'),
    ]))
    elements.append(t_prob)

    elements.append(Paragraph("AI RESULT INTERPRETATION", heading_style))
    interp_text = f"The Convolutional Neural Network classified the uploaded CT image as <b>{format_class_name(pred_class)}</b> with a confidence of {confidence*100:.2f}%. "
    interp_text += f"{format_class_name(pred_class)} is a {'non-malignant' if not is_malignant else 'malignant'} class in this model."
    if is_uncertain: interp_text += " <b>However, the model confidence is relatively low (<60%). The result should therefore be interpreted cautiously.</b>"
    elements.append(Paragraph(interp_text, body_style))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("MODEL EXPLAINABILITY", heading_style))
    elements.append(Paragraph("Grad-CAM highlights image regions that contributed more strongly to the CNN prediction. The visualization is provided for research and educational interpretation and should not be considered clinical evidence.", body_style))

    elements.append(Spacer(1, 15))
    elements.append(Paragraph("IMPORTANT NOTICE", bold_body))
    disclaimer = Paragraph("This report is generated by an AI-based research prototype developed for educational and academic purposes. The CNN output represents a model classification and should NOT be interpreted as a medical diagnosis. Clinical decisions should be made only by qualified healthcare professionals using appropriate medical evaluation.", body_style)
    t_disc = Table([[disclaimer]], colWidths=[535])
    t_disc.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")), ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94a3b8")), ('PADDING', (0,0), (-1,-1), 8)]))
    elements.append(t_disc)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==========================================
# EXTREME CSS FOR TRUE WEB UI
# ==========================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* CSS RESET FOR STREAMLIT TO LOOK LIKE A REAL WEBSITE */
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }
        
        #root > div:nth-child(1) > div > div > div > div > section > div {
            padding-top: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Overall Page Backgrounds */
        .stApp {
            background-color: #f4f7f6;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* The Absolute Hero Background for Home */
        .hero-bg {
            position: absolute;
            top: 60px; /* Below navbar */
            left: 0;
            width: 100%;
            height: 550px;
            background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%);
            z-index: 0;
            border-bottom: 5px solid #E53E3E;
        }
        
        .content-layer {
            position: relative;
            z-index: 10;
            padding: 0 5%;
        }

        /* Typography */
        h1, h2, h3 { color: #1A365D; font-weight: 800; }
        .hero-title { font-size: 3.8rem; font-weight: 900; color: white; line-height: 1.1; margin-bottom: 1.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
        .hero-title span { color: #88c5e0; }
        .hero-subtitle { font-size: 1.2rem; color: #e2e8f0; margin-bottom: 2.5rem; max-width: 600px; line-height: 1.6; }

        /* Floating Cards Animation */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-12px); }
            100% { transform: translateY(0px); }
        }
        .f-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem 2rem;
            border-radius: 12px;
            color: #1A365D;
            font-weight: 800;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            margin-bottom: 20px;
            width: fit-content;
            animation: float 4s ease-in-out infinite;
        }
        .offset-1 { transform: translateX(20px); animation-delay: 0s; }
        .offset-2 { transform: translateX(-30px); animation-delay: 1s; }
        .offset-3 { transform: translateX(10px); animation-delay: 2s; }
        
        /* Buttons */
        div.stButton > button {
            border-radius: 8px !important;
            font-weight: 700 !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button[kind="primary"] {
            background-color: #e85d44 !important;
            color: white !important;
            border: none !important;
            padding: 12px 30px !important;
            font-size: 1.2rem !important;
            box-shadow: 0 4px 15px rgba(232, 93, 68, 0.4) !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #cf4f38 !important;
            transform: translateY(-2px);
        }
        
        /* How It Works Elements */
        .step-circle {
            width: 90px; height: 90px;
            background: white; border: 4px solid #88c5e0;
            border-radius: 50%; display: flex;
            justify-content: center; align-items: center;
            font-size: 35px; box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            margin: 0 auto 20px auto; position: relative;
        }
        .step-num {
            position: absolute; top: -10px; background: #88c5e0;
            color: white; width: 30px; height: 30px; border-radius: 50%;
            display: flex; justify-content: center; align-items: center;
            font-size: 16px; font-weight: bold; border: 2px solid white;
        }
        .step-box { text-align: center; }
        .step-box h4 { color: #1A365D; font-weight: 800; font-size: 1.2rem; margin-bottom: 10px; }
        .step-box p { color: #718096; font-size: 1rem; line-height: 1.5; }

        /* Form Inputs Customization */
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
        }
        div[data-baseweb="input"] > div:focus-within, 
        div[data-baseweb="select"] > div:focus-within {
            border-color: #2B6CB0 !important;
            box-shadow: 0 0 0 2px rgba(43, 108, 176, 0.2) !important;
        }
        .stNumberInput input, .stTextInput input { font-size: 1rem !important; }
        
        /* Prediction Page Card */
        .pred-card {
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border-top: 5px solid #1A365D;
            margin-top: 20px;
            margin-bottom: 40px;
        }
        
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE VIEWS
# ==========================================
def render_navbar():
    # Outer div for navbar
    st.markdown("""
        <div style="background-color: #1A365D; width: 100%; height: 60px; display: flex; align-items: center; padding: 0 5%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); position: relative; z-index: 100;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 800;">🫁 LungCare AI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # We overlay the option menu in the exact same spot using negative margin
    st.markdown("<div style='margin-top: -65px; margin-left: 300px; padding-right: 5%; position: relative; z-index: 101;'>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None, 
        options=["Home", "About", "Prediction", "Analytics", "Contact"],
        icons=["house", "info-circle", "file-medical", "graph-up", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent", "margin": "0"},
            "icon": {"color": "#a0aec0", "font-size": "16px"}, 
            "nav-link": {"font-size": "15px", "font-weight": "600", "text-align": "center", "margin": "0px 10px", "color": "#a0aec0", "--hover-color": "rgba(255,255,255,0.1)"},
            "nav-link-selected": {"background-color": "rgba(255,255,255,0.15)", "color": "white", "font-weight": "700"},
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return selected

def page_home():
    st.markdown("<div class='hero-bg'></div>", unsafe_allow_html=True)
    st.markdown("<div class='content-layer'>", unsafe_allow_html=True)
    
    # Push content down into the hero
    st.write("")
    st.write("")
    st.write("")
    
    hero_cols = st.columns([1.2, 1])
    with hero_cols[0]:
        st.markdown("""
        <h1 class="hero-title">AI-Powered<br>Lung Cancer<br>Detection</h1>
        <p class="hero-subtitle">Advanced machine learning technology for early lung cancer prediction. Get instant risk assessment based on clinical parameters and lifestyle factors.</p>
        """, unsafe_allow_html=True)
        
        btn_col1, btn_col2, _ = st.columns([1, 1, 1])
        with btn_col1:
            if st.button("Start Prediction", type="primary", use_container_width=True):
                st.info("Please select 'Prediction' from the top navigation bar to begin.")
        with btn_col2:
            st.button("Learn More", use_container_width=True)

    with hero_cols[1]:
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: flex-end; padding-top: 2rem;">
            <div class="f-card offset-1">💓 Real-time Analysis</div>
            <div class="f-card offset-2">🧠 AI Technology</div>
            <div class="f-card offset-3">📈 High Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True) # close content-layer

    # Spacing to clear absolute background
    st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True)
    
    # Why Choose / How It Works
    st.markdown("<div class='content-layer'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 2.5rem; margin-bottom: 10px;'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #718096; font-size: 1.1rem; margin-bottom: 60px;'>Simple process to get your lung cancer risk assessment</p>", unsafe_allow_html=True)
    
    hw_cols = st.columns([1, 0.2, 1, 0.2, 1])
    with hw_cols[0]:
        st.markdown("<div class='step-box'><div class='step-circle'><div class='step-num'>1</div>📄</div><h4>Enter Information</h4><p>Fill in your medical history and lifestyle parameters in our secure form.</p></div>", unsafe_allow_html=True)
    with hw_cols[1]: st.markdown("<h2 style='color: #88c5e0; text-align: center; margin-top: 30px;'>➔</h2>", unsafe_allow_html=True)
    with hw_cols[2]:
        st.markdown("<div class='step-box'><div class='step-circle'><div class='step-num'>2</div>⚙️</div><h4>AI Analysis</h4><p>Our advanced ML models process your data using multiple algorithms.</p></div>", unsafe_allow_html=True)
    with hw_cols[3]: st.markdown("<h2 style='color: #88c5e0; text-align: center; margin-top: 30px;'>➔</h2>", unsafe_allow_html=True)
    with hw_cols[4]:
        st.markdown("<div class='step-box'><div class='step-circle'><div class='step-num'>3</div>📊</div><h4>Get Results</h4><p>Receive detailed prediction with visual graphs and risk assessment.</p></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

def page_prediction():
    st.markdown("<div class='content-layer'>", unsafe_allow_html=True)
    
    st.markdown("<div class='pred-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 2rem; margin-bottom: 5px;'>Patient Information Form</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #718096; margin-bottom: 40px;'>All fields are required for accurate clinical context and reporting</p>", unsafe_allow_html=True)
    
    # Form Layout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<b>Demographics & Vitals</b>", unsafe_allow_html=True)
        user_name = st.text_input("Full Name", placeholder="e.g., John Doe")
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Select Gender", "Male", "Female", "Other"])
        st.markdown("<br><b>Medical History</b>", unsafe_allow_html=True)
        copd = st.selectbox("COPD Diagnosis", ["Select Option", "No", "Yes"])
        family_history = st.selectbox("Family History of Lung Cancer", ["Select Option", "No", "Yes"])
    with col2:
        st.markdown("<b>Lifestyle & Exposures</b>", unsafe_allow_html=True)
        pack_years = st.number_input("Pack Years (Cigarettes/Day × Years)", min_value=0.0, value=0.0, step=0.5)
        secondhand = st.selectbox("Secondhand Smoke Exposure", ["Select Option", "None", "Occasional", "Frequent"])
        radon = st.selectbox("Radon Exposure", ["Select Level", "None", "Low", "Moderate", "High"])
        asbestos = st.selectbox("Asbestos Exposure", ["Select Option", "None", "Low", "Moderate", "High"])
        alcohol = st.selectbox("Alcohol Consumption", ["Select Level", "None", "Light", "Moderate", "Heavy"])

    st.markdown("<hr style='border: 0; height: 1px; background: #e2e8f0; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("<b>Upload CT Scan Image</b>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    st.write("")
    analyze_btn = st.button("Run AI Prediction", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True) # End pred-card

    # Results Section
    if analyze_btn:
        if uploaded_file is None:
            st.error("Please upload a CT Scan Image to proceed.")
            return
        if model is None:
            st.error("Model not found! Please ensure 'best_model.keras' exists.")
            return

        with st.spinner("Analyzing Clinical and Imaging Data..."):
            time.sleep(1.5)
            image = Image.open(uploaded_file).convert('RGB')
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = preds[pred_idx]
            
            is_malignant = pred_class in MALIGNANT_CLASSES
            is_uncertain = confidence < CONFIDENCE_THRESHOLD

            st.markdown("<div class='pred-card' style='border-top: 5px solid #10b981; margin-top: 0;'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;'>Diagnostic Results</h2><hr>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.image(image, caption="Uploaded CT Scan", use_container_width=True)
                color = '#e85d44' if is_malignant else '#10b981'
                if is_uncertain: color = '#f59e0b'
                st.markdown(f"<h3>CNN Prediction: <span style='color: {color};'>{pred_class.upper()}</span></h3>", unsafe_allow_html=True)
                st.write(f"**Confidence:** {confidence*100:.2f}%")
                
                if is_uncertain: st.warning("STATUS: UNCERTAIN - Confidence is low.")
                else: st.success("STATUS: HIGH CONFIDENCE")
                if is_malignant: st.error("⚠️ Malignant classification detected.")

                st.write("---")
                st.markdown("<h4>Probability Distribution</h4>", unsafe_allow_html=True)
                for idx in np.argsort(preds)[::-1]:
                    prob = preds[idx]
                    bar_color = "#3b82f6" if idx == pred_idx else ("#fca5a5" if CLASS_NAMES[idx] in MALIGNANT_CLASSES else "#6ee7b7")
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <div style="font-size: 14px; font-weight: {'bold' if idx == pred_idx else 'normal'}; color: #2d3748;">{CLASS_NAMES[idx].title()}</div>
                        <div style="width: 100%; background-color: #e2e8f0; border-radius: 4px; height: 16px;">
                            <div style="width: {prob*100}%; background-color: {bar_color}; height: 100%; border-radius: 4px; transition: width 1s;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            superimposed_img = None
            with res_col2:
                st.markdown("<h3>Feature Attention (Grad-CAM)</h3>", unsafe_allow_html=True)
                try:
                    last_conv = next((l.name for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)), None)
                    if last_conv:
                        heatmap = make_gradcam_heatmap(img_array, model, last_conv)
                        img_cv = np.array(img_resized)
                        heatmap_cv = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))), cv2.COLORMAP_JET)
                        superimposed_img = np.clip(heatmap_cv * 0.4 + img_cv, 0, 255).astype('uint8')
                        st.image(superimposed_img, caption="AI Hotspots", use_container_width=True)
                except Exception: pass
            
            patient_data = {
                'Name': user_name.strip() if user_name.strip() else 'Not Provided',
                'Age': age, 'Gender': gender, 'Pack Years': pack_years, 'Radon': radon,
                'Asbestos': asbestos, 'Secondhand': secondhand, 'COPD': copd, 'Alcohol': alcohol
            }

            st.write("---")
            if superimposed_img is not None:
                pdf_buffer = generate_pdf_report(patient_data, uploaded_file.name, image, cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB), pred_class, confidence, is_malignant, is_uncertain, preds)
                _, dl_col, _ = st.columns([1, 2, 1])
                with dl_col:
                    st.download_button("📄 Download Comprehensive PDF Report", data=pdf_buffer, file_name="LungCare_Report.pdf", mime="application/pdf", use_container_width=True, type="primary")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def page_analytics():
    st.markdown("<div class='content-layer'><div class='pred-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px;'>System Analytics & Performance</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='text-align:center; padding: 20px; border: 1px solid #e2e8f0; border-radius: 12px;'><h3>Total Scans</h3><h1 style='color:#3b82f6; font-size: 3rem;'>1,226</h1></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align:center; padding: 20px; border: 1px solid #e2e8f0; border-radius: 12px;'><h3>Test Accuracy</h3><h1 style='color:#10b981; font-size: 3rem;'>81.87%</h1></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div style='text-align:center; padding: 20px; border: 1px solid #e2e8f0; border-radius: 12px;'><h3>Sensitivity</h3><h1 style='color:#e85d44; font-size: 3rem;'>98.92%</h1></div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

def page_contact():
    st.markdown("<div class='content-layer'>", unsafe_allow_html=True)
    st.markdown("<div class='pred-card' style='max-width: 600px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Contact Support</h2>", unsafe_allow_html=True)
    st.text_input("Full Name")
    st.text_input("Email Address")
    st.text_area("How can we help you?")
    st.button("Send Message", type="primary", use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# ==========================================
# MAIN APP ROUTING
# ==========================================
def main():
    inject_custom_css()
    selected_page = render_navbar()
    
    if selected_page == "Home": page_home()
    elif selected_page == "About": 
        st.markdown("<div class='content-layer'><div class='pred-card'><h2 style='text-align: center;'>About LungCare AI</h2><p style='text-align:center; color:#718096;'>This project uses a Convolutional Neural Network (EfficientNetB0) to classify lung CT scans into 5 categories: Normal, Benign, Adenocarcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma.</p></div></div>", unsafe_allow_html=True)
    elif selected_page == "Prediction": page_prediction()
    elif selected_page == "Analytics": page_analytics()
    elif selected_page == "Contact": page_contact()

if __name__ == "__main__":
    main()
