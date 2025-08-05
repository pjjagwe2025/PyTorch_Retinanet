import torch

# Basic training configuration
BATCH_SIZE = 4  # Adjust based on GPU memory
RESIZE_TO = 640
NUM_EPOCHS = 40
NUM_WORKERS = 4

# Set device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Custom dataset directories (after re-organizing with /images and /annotations folders)
TRAIN_DIR = 'custom_data/train/images'
VALID_DIR = 'custom_data/valid/images'
TEST_DIR  = 'custom_data/test/images'  # Optional: add this if you're using it in eval/inference

# Custom classes (first index is background)
CLASSES = [
    '__background__', 'plant', 'weed'
]

NUM_CLASSES = len(CLASSES)

# Visualization toggle
VISUALIZE_TRANSFORMED_IMAGES = False

# Output directory
OUT_DIR = 'outputs'
