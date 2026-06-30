import torch

class Config:
    # Video & Frame Processing
    FRAME_WIDTH = 224
    FRAME_HEIGHT = 224
    SEQUENCE_LENGTH = 16  # Number of frames for spatiotemporal analysis
    CHANNELS = 3
    
    # Model Architecture
    PATCH_SIZE = 16
    EMBED_DIM = 192
    NUM_HEADS = 3
    NUM_LAYERS = 4
    NUM_CLASSES = 2  # [Real, Fake]
    
    # Inference Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIDENCE_THRESHOLD = 0.65
    
    # Biometrics Thresholds
    EAR_THRESHOLD = 0.22  # Eye Aspect Ratio below this indicates closed eye