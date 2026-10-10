"""
LungCare AI  —  server.py
Flask backend for the real website.
Key fixes:
  - use_reloader=False  (prevents watchdog killing live prediction requests)
  - threaded=True       (handles concurrent requests)
  - try/except on /predict so any error returns JSON, not HTML
  - model pre-loaded before Flask starts
"""
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2, os, io, uuid, base64, datetime
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Image as RLImage, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

app = Flask(__name__)
CORS(app)

# ── Config ──────────────────────────────────────────────────────
MODEL_PATH = "results/best_model.keras"
CLASS_NAMES = [
    'Benign cases', 'Normal cases',
    'adenocarcinoma', 'large cell carcinoma', 'squamous cell carcinoma'
]
MALIGNANT      = {'adenocarcinoma', 'large cell carcinoma', 'squamous cell carcinoma'}
CONF_THRESHOLD = 0.60

# ── Model (loaded once at startup) ──────────────────────────────
_model = None
def get_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        print("Loading CNN model …")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded OK")
    return _model

# ── Grad-CAM ────────────────────────────────────────────────────
def make_gradcam(img_batch, model):
    try:
        last_conv = next(
            (l.name for l in reversed(model.layers)
             if isinstance(l, tf.keras.layers.Conv2D)), None
        )
        if not last_conv:
            return None
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_batch)
            cls_ch = preds[:, tf.argmax(preds[0])]
        grads   = tape.gradient(cls_ch, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        mx      = tf.math.reduce_max(heatmap)
        heatmap = heatmap / (mx + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_gradcam(orig_rgb, heatmap):
    h, w   = orig_rgb.shape[:2]
    heat   = cv2.resize(heatmap, (w, h))
    heat   = np.uint8(255 * heat)
    heat   = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    result = np.clip(heat * 0.4 + orig_rgb, 0, 255).astype('uint8')
    return result  # RGB

def arr_to_b64(arr_rgb):
    buf = io.BytesIO()
    Image.fromarray(arr_rgb).save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()

# ── Pre-process uploaded image ───────────────────────────────────
def preprocess_bytes(img_bytes):
    pil    = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img224 = pil.resize((224, 224))
    arr    = tf.keras.preprocessing.image.img_to_array(img224)
    batch  = np.expand_dims(arr, 0)
    return pil, img224, batch

# ── PDF Builder ─────────────────────────────────────────────────
def build_pdf(patient, fname, orig_pil, gradcam_rgb,
              pred_cls, conf, is_mal, is_unc, probs):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=30, leftMargin=30,
                            topMargin=30,  bottomMargin=30)
    els = []
    SS  = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, parent=SS['Normal'], **kw)

    h2   = ps('h2', fontSize=11, fontName='Helvetica-Bold',
               textColor=colors.HexColor('#1e293b'),
               spaceBefore=10, spaceAfter=4)
    body = ps('body', fontSize=9,
               textColor=colors.HexColor('#334155'), leading=12)

    now = datetime.datetime.now()
    rid = str(uuid.uuid4())[:8].upper()

    # Header
    hdr = Table([[
        Paragraph('<b>LungCare AI</b><br/>'
                  'Lung CT Classification — BE Final Year Project', body),
        Paragraph(f'<font color="#0284c7"><b>RESEARCH PROTOTYPE</b></font>'
                  f'<br/>ID: {rid}'
                  f'<br/>{now.strftime("%Y-%m-%d  %H:%M")}', body)
    ]], colWidths=[350, 185])
    hdr.setStyle(TableStyle([
        ('VALIGN',      (0,0), (-1,-1), 'TOP'),
        ('LINEBELOW',   (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
    ]))
    els.append(hdr)

    # Patient info
    els.append(Paragraph('PATIENT INFORMATION', h2))
    rows = [
        ['Name:',         patient.get('name','—'),
         'Age / Gender:', f"{patient.get('age','—')} / {patient.get('gender','—')}"],
        ['Occupation:',   patient.get('occupation','—'),
         'City/Region:',  patient.get('city','—')],
        ['Pack Years:',   patient.get('pack_years','—'),
         'Radon:',        patient.get('radon','—')],
        ['Asbestos:',     patient.get('asbestos','—'),
         'Secondhand:',   patient.get('secondhand','—')],
        ['COPD:',         patient.get('copd','—'),
         'Alcohol:',      patient.get('alcohol','—')],
        ['Family Hx:',    patient.get('family_hx','—'),
         'Prior Chemo:',  patient.get('chemo','—')],
        ['Prev. Lung:',   patient.get('prev_conditions','—'),
         'Radiation:',    patient.get('radiation','—')],
    ]
    pt = Table(rows, colWidths=[80, 170, 80, 205])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX',        (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',   (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('PADDING',    (0,0), (-1,-1), 5),
    ]))
    els.append(pt)

    # Images
    def pil_rl(p, w=2.3, h=2.3):
        b = io.BytesIO()
        p.save(b, 'JPEG')
        b.seek(0)
        return RLImage(b, width=w*inch, height=h*inch)

    els.append(Paragraph('CT IMAGE ANALYSIS', h2))
    img_t = Table([
        [Paragraph('<b>Original CT Scan</b>',
                   ps('c1', alignment=1, fontSize=9)),
         Paragraph('<b>Grad-CAM Feature Map</b>',
                   ps('c2', alignment=1, fontSize=9))],
        [pil_rl(orig_pil),
         pil_rl(Image.fromarray(gradcam_rgb))]
    ], colWidths=[267, 268])
    img_t.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOX',   (0,0), (0,-1), 1, colors.HexColor('#e2e8f0')),
        ('BOX',   (1,0), (1,-1), 1, colors.HexColor('#e2e8f0')),
    ]))
    els.append(img_t)

    # Result
    els.append(Paragraph('CNN CLASSIFICATION RESULT', h2))
    sc = colors.HexColor('#ef4444' if is_mal else '#10b981')
    if is_unc: sc = colors.HexColor('#f59e0b')
    bg = colors.HexColor('#fee2e2' if is_mal else '#d1fae5')
    if is_unc: bg = colors.HexColor('#fef3c7')
    rt = Table([
        ['Predicted Class:', pred_cls.title(),
         'Confidence:', f'{conf*100:.2f}%'],
        ['Status:',
         'UNCERTAIN' if is_unc else 'HIGH CONFIDENCE',
         'Type:', 'MALIGNANT' if is_mal else 'NON-MALIGNANT'],
    ], colWidths=[90, 177, 70, 198])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('BOX',        (0,0), (-1,-1), 2, sc),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',   (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTNAME',   (1,0), (1,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (1,0), (1,-1), sc),
        ('PADDING',    (0,0), (-1,-1), 7),
    ]))
    els.append(rt)

    # Probabilities
    els.append(Paragraph('CLASS PROBABILITIES', h2))
    pb = [['Class', 'Probability']] + [
        [CLASS_NAMES[i].title(), f'{probs[i]*100:.2f}%']
        for i in np.argsort(probs)[::-1]
    ]
    pt2 = Table(pb, colWidths=[350, 185])
    pt2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID',       (0,0), (-1,-1), .5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f8fafc')),
        ('FONTNAME',   (0,1), (-1,1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('PADDING',    (0,0), (-1,-1), 5),
    ]))
    els.append(pt2)

    # Interpretation
    els.append(Paragraph('INTERPRETATION', h2))
    interp = (f'The CNN classified the CT scan as <b>{pred_cls.title()}</b> '
              f'with {conf*100:.2f}% confidence. '
              f'This is a <b>{"non-" if not is_mal else ""}malignant</b> class.')
    if is_unc:
        interp += ' <b>Confidence is below 60% — treat result as UNCERTAIN.</b>'
    els.append(Paragraph(interp, body))

    els.append(Spacer(1, 10))
    disc = Paragraph(
        '<b>DISCLAIMER:</b> This report is produced by an AI research prototype '
        'for educational purposes only. It must NOT be used for clinical diagnosis. '
        'Always consult a qualified medical professional.', body)
    td = Table([[disc]], colWidths=[535])
    td.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX',        (0,0), (-1,-1), 1, colors.HexColor('#94a3b8')),
        ('PADDING',    (0,0), (-1,-1), 8),
    ]))
    els.append(td)

    doc.build(els)
    buf.seek(0)
    return buf

# ── Routes ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = get_model()
        if model is None:
            return jsonify({'error': 'Model not found at results/best_model.keras'}), 500

        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({'error': 'No image uploaded'}), 400

        # Read bytes once (avoids stream issues)
        img_bytes            = file.read()
        pil, img224, batch   = preprocess_bytes(img_bytes)
        orig_arr             = np.array(img224)   # RGB uint8

        preds    = model.predict(batch, verbose=0)[0]
        idx      = int(np.argmax(preds))
        pred_cls = CLASS_NAMES[idx]
        conf     = float(preds[idx])
        is_mal   = pred_cls in MALIGNANT
        is_unc   = conf < CONF_THRESHOLD

        heatmap     = make_gradcam(batch, model)
        gradcam_rgb = overlay_gradcam(orig_arr, heatmap) if heatmap is not None else orig_arr

        return jsonify({
            'pred_class'    : pred_cls,
            'pred_class_fmt': pred_cls.title(),
            'confidence'    : round(conf * 100, 2),
            'is_malignant'  : is_mal,
            'is_uncertain'  : is_unc,
            'probabilities' : {
                CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            'orig_b64'      : arr_to_b64(orig_arr),
            'gradcam_b64'   : arr_to_b64(gradcam_rgb),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        model = get_model()
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({'error': 'No image uploaded'}), 400

        img_bytes          = file.read()
        pil, img224, batch = preprocess_bytes(img_bytes)
        orig_arr           = np.array(img224)

        preds    = model.predict(batch, verbose=0)[0]
        idx      = int(np.argmax(preds))
        pred_cls = CLASS_NAMES[idx]
        conf     = float(preds[idx])
        is_mal   = pred_cls in MALIGNANT
        is_unc   = conf < CONF_THRESHOLD

        heatmap     = make_gradcam(batch, model)
        gradcam_rgb = overlay_gradcam(orig_arr, heatmap) if heatmap is not None else orig_arr

        form = request.form
        patient = {k: (form.get(k, '') or '—')
                   for k in ['name','age','gender','occupation','city',
                              'pack_years','radon','asbestos','secondhand',
                              'alcohol','family_hx','copd','prev_conditions',
                              'chemo','radiation']}

        pdf = build_pdf(patient, file.filename,
                        pil, gradcam_rgb,
                        pred_cls, conf, is_mal, is_unc, preds)

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(pdf, mimetype='application/pdf',
                         as_attachment=True,
                         download_name=f'LungCare_Report_{ts}.pdf')

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Entry point ─────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    # Pre-load model BEFORE Flask starts (avoids first-request delay & reloader clash)
    get_model()
    print("\n" + "="*50)
    print("  LungCare AI — Real Website")
    print("  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    # IMPORTANT: use_reloader=False prevents watchdog from killing live requests
    app.run(debug=False, port=5000, host='0.0.0.0',
            use_reloader=False, threaded=True)
