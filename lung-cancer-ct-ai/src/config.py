import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
GRADCAM_DIR = os.path.join(OUTPUT_DIR, 'gradcam')

# Default Model Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
CONFIDENCE_THRESHOLD = 0.60

# Model configurations
SUPPORTED_MODELS = [
    "Custom CNN",
    "EfficientNetB0",
    "ResNet50",
    "MobileNetV2",
    "Compare All"
]

# File formats supported
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.dcm')

def create_dirs():
    """Ensure all required directories exist."""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(GRADCAM_DIR, exist_ok=True)

# Run directory creation on import
create_dirs()
