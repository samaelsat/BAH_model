"""Global constants for BAH multimodal training."""

# Metrics Constants
MACRO_F1 = "MACRO_F1"
W_F1 = "W_F1"
CL_ACC = "CL_ACC"
CFUSE_MARIX = "CONFUSION_MATRIX"

# Additional A/H metrics
F1_POS = "F1_POS"
F1_NEG = "F1_NEG"
AP_POS = "Average_precision_POS"

# Model Constants
NUM_CLASSES = 2

# Video Constants
VIDEO_FRAME_SIZE = 250
NUM_FRAMES = 32
VIDEO_CHANNELS = 3

# Audio Constants
AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_DURATION = 10

# Text Constants
TEXT_MAX_LENGTH = 128

# Fusion Constants
FUSION_HIDDEN_DIM = 512
FUSION_NUM_HEADS = 8
FUSION_NUM_LAYERS = 4

# Training Constants
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 100
WARMUP_EPOCHS = 5

# GCS Paths
GCS_BUCKET = "gs://ah-classify/data/bifurcated/"
GCS_TRAIN_PATH = "train/"
GCS_VAL_PATH = "val/"
GCS_TEST_PATH = "test/"
GCS_TRANSCRIPT_PATH = "gs://ah-classify/data/video_annotation_transcript.yaml"
GCS_SPLIT_PATHS = {
    "train": {
        0: "gs://ah-classify/data/split/train_0.txt",
        1: "gs://ah-classify/data/split/train_1.txt",
    },
    "val": {
        0: "gs://ah-classify/data/split/val_0.txt",
        1: "gs://ah-classify/data/split/val_1.txt",
    },
    "test": {
        0: "gs://ah-classify/data/split/test_0.txt",
        1: "gs://ah-classify/data/split/test_1.txt",
    },
}
