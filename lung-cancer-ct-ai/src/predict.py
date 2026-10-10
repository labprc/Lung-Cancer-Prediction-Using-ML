import argparse
import os
import tensorflow as tf
import numpy as np
from src.preprocessing import preprocess_image
from src.utils import load_json
from src.gradcam import get_last_conv_layer_name, make_gradcam_heatmap, save_and_display_gradcam
from src.config import MODELS_DIR, IMAGE_SIZE, CONFIDENCE_THRESHOLD

def predict_image(image_path):
    model_path = os.path.join(MODELS_DIR, 'best_model.keras')
    if not os.path.exists(model_path):
        print("No trained model found. Please train a model first.")
        return None
        
    class_names = load_json('class_names.json')
    if not class_names:
        print("Class names not found. Training might be incomplete.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    img_array, _ = preprocess_image(image_path, IMAGE_SIZE)
    preds = model.predict(img_array)[0]
    
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = preds[pred_index]
    
    result = {
        'predicted_class': pred_class,
        'confidence': float(confidence),
        'probabilities': {class_names[i]: float(preds[i]) for i in range(len(class_names))},
        'low_confidence': confidence < CONFIDENCE_THRESHOLD
    }
    
    # Grad-CAM
    layer_name = get_last_conv_layer_name(model)
    if layer_name:
        heatmap = make_gradcam_heatmap(img_array, model, layer_name, pred_index)
        if heatmap is not None:
            cam_path, orig, hm, overlay = save_and_display_gradcam(image_path, heatmap)
            result['gradcam_path'] = cam_path
            
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help="Path to CT scan image")
    args = parser.parse_args()
    
    res = predict_image(args.image_path)
    if res:
        print(f"\nPredicted Class: {res['predicted_class']}")
        print(f"Confidence: {res['confidence']*100:.2f}%\n")
        print("Class Probabilities:")
        for cls, prob in res['probabilities'].items():
            print(f"{cls}: {prob*100:.2f}%")
        
        if res.get('low_confidence'):
            print("\nWARNING: Low-confidence model prediction.")
            
        if 'gradcam_path' in res:
            print(f"\nGrad-CAM saved to: {res['gradcam_path']}")
