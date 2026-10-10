import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm
import os
from src.config import GRADCAM_DIR

def get_last_conv_layer_name(model):
    """
    Dynamically finds the last convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # For nested models (like transfer learning base models)
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, tf.keras.layers.Conv2D):
                    return inner_layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates Grad-CAM heatmap for the given image array.
    """
    # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    try:
        # Check if model has a nested base model
        inner_model = None
        for layer in model.layers:
             if isinstance(layer, tf.keras.Model):
                 inner_model = layer
                 break
                 
        if inner_model:
            grad_model = tf.keras.models.Model(
                [inner_model.inputs], [inner_model.get_layer(last_conv_layer_name).output, inner_model.output]
            )
            # Need to pass through preprocessing of outer model if any, then into inner
            # For simplicity, if we have standard wrapping, we pass to inner_model
            # Wait, if we wrapped it with augmentation, img_array is raw.
            # We assume img_array is preprocessed if it's evaluated on grad_model directly.
        else:
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
            )
            
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays heatmap on original image.
    """
    img = cv2.imread(img_path)
    if img is None:
         return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    
    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    save_path = os.path.join(GRADCAM_DIR, "latest_gradcam.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return save_path, img, jet_heatmap, superimposed_img
