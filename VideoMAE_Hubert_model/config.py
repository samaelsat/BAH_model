"""
BAH A/H Recognition - Configuration
ABAW10 @ CVPR 2026 Challenge
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # VideoMAE backbone (processes 2x16-frame clips for 32 total frames)
    video_model_name: str = "MCG-NJU/videomae-base"
    # HuBERT audio backbone
    audio_model_name: str = "facebook/hubert-base-ls960"

    # Shared feature dimension (both video+audio projected here)
    d_model: int = 768
    # Cross-modal transformer
    nhead: int = 8
    num_cross_layers: int = 2
    dropout: float = 0.3

    # LSTM for temporal binary classification
    lstm_hidden: int = 512
    lstm_layers: int = 2
    lstm_bidirectional: bool = True

    num_classes: int = 2

    # Optionally freeze backbones for faster early training
    freeze_video_backbone: bool = False
    freeze_audio_backbone: bool = False


@dataclass
class DataConfig:
    # GCS path — bucket: ah-classify, path: data/bifurcated
    gcs_data_path: str = "gs://ah-classify/data/bifurcated"
    local_data_dir: str = "/tmp/bah_data"

    # 32 frames total split into 2 clips of 16 for VideoMAE
    num_frames: int = 32
    num_frames_per_clip: int = 16   # VideoMAE base is pretrained on 16 frames

    # VideoMAE native resolution
    frame_size: int = 224           # user specified 225 → VideoMAE requires 224

    # Audio
    sample_rate: int = 16000        # HuBERT expects 16kHz mono
    max_audio_len_sec: float = 90.0

    # Augmentation
    audio_noise_std: float = 0.005  # Gaussian noise injection for training


@dataclass
class TrainConfig:
    # Outputs
    output_dir: str = "/tmp/bah_outputs"
    gcs_output_path: str = "gs://ah-classify/outputs"

    batch_size: int = 4             # keep low — video models are memory heavy
    num_epochs: int = 30
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 5

    # Differential LRs: small for pretrained backbones, larger for new heads
    backbone_lr: float = 5e-6
    lr: float = 2e-5                # cross-modal transformer, LSTM, classifier
    weight_decay: float = 1e-4

    warmup_steps: int = 100
    gradient_clip: float = 1.0
    mixed_precision: bool = True    # AMP on GPU

    num_workers: int = 4
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
