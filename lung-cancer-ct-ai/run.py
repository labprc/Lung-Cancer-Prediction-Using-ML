import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import cv2
import sys
from src.predict import predict_image

def select_image_and_predict():
    file_path = None
    
    # If user provided a path via command line, use it
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Otherwise, open file dialog and force it to the front
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Force window to the front
        
        file_path = filedialog.askopenfilename(
            title="Select a CT Scan Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")]
        )
        root.destroy()
    
    if not file_path:
        print("No image selected. Exiting.")
        return
        
    print(f"Analyzing image: {file_path}")
    print("Please wait...")
    
    # Run prediction
    result = predict_image(file_path)
    
    if not result:
        messagebox.showerror("Error", "Prediction failed. Ensure the model is trained and exists in the models/ folder.")
        return
        
    # Print results to console
    print("\n--- PREDICTION RESULTS ---")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("\nAll Probabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob*100:.2f}%")
        
    if result.get('low_confidence'):
        print("\nWARNING: Low-confidence prediction.")
        
    # Display results using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('LungCT-AI Prediction Results')
    
    # Original Image
    orig_img = cv2.imread(file_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_img)
    axes[0].set_title("Original CT Scan")
    axes[0].axis('off')
    
    # Grad-CAM Image
    if 'gradcam_path' in result:
        cam_img = cv2.imread(result['gradcam_path'])
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(cam_img)
        axes[1].set_title("Grad-CAM AI Attention")
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Grad-CAM Not Available', horizontalalignment='center', verticalalignment='center')
        axes[1].axis('off')
        
    # Add text info below images
    info_text = f"Predicted: {result['predicted_class']} ({result['confidence']*100:.1f}%)\n\n"
    info_text += "Probabilities:\n"
    for cls, prob in result['probabilities'].items():
        info_text += f"{cls}: {prob*100:.1f}%\n"
        
    plt.figtext(0.5, 0.02, info_text, wrap=True, horizontalalignment='center', fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                
    plt.tight_layout(rect=[0, 0.15, 1, 1]) # Leave space at bottom for text
    plt.show()

if __name__ == "__main__":
    select_image_and_predict()
