from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
from werkzeug.utils import secure_filename
from src.predict import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs/gradcam', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
        
    if file:
        filename = secure_filename(file.filename)
        # Add timestamp to prevent caching issues
        unique_filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Run prediction
            result = predict_image(filepath)
            
            if not result:
                return jsonify({'error': 'Prediction failed. Ensure model is trained.'})
                
            # Prepare response
            response = {
                'predicted_class': result['predicted_class'],
                'confidence': f"{result['confidence']*100:.2f}%",
                'probabilities': {k: f"{v*100:.2f}%" for k, v in result['probabilities'].items()},
                'original_image': f"/{filepath}",
                'gradcam_image': f"/outputs/gradcam/latest_gradcam.jpg?t={int(time.time())}" # cache buster
            }
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)})

# Route to serve the generated outputs (Grad-CAM)
@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory('outputs', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
