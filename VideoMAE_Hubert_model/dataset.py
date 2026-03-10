"""
BAH A/H Recognition - Dataset
ABAW10 @ CVPR 2026 Challenge

Data structure expected:
  {local_data_dir}/
    train/
      0/   <- no A/H videos (*.mp4)
      1/   <- A/H present  (*.mp4)
    val/
      0/
      1/
    test/
      0/
      1/
"""

import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T
import torchaudio
import cv2


# ────────────────────────────────────────────────────────────────────────────────
# Video transforms
# ────────────────────────────────────────────────────────────────────────────────

def build_video_transform(frame_size: int, is_training: bool):
    """Per-frame transform pipeline."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if is_training:
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.Resize((frame_size, frame_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((frame_size, frame_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────

class BAHVideoDataset(Dataset):
    """
    Loads .mp4 videos from bifurcated split folders (0/, 1/).
    For each video:
      - Extracts `num_frames` frames using random temporal sampling (training)
        or uniform sampling (eval/test).
      - Extracts mono audio at 16kHz using ffmpeg.
      - Optionally injects Gaussian noise into audio (training only).
    """

    def __init__(
        self,
        data_dir: str,
        split: str,                  # 'train' | 'val' | 'test'
        num_frames: int = 32,
        frame_size: int = 224,
        sample_rate: int = 16000,
        max_audio_len_sec: float = 90.0,
        audio_noise_std: float = 0.005,
        is_training: bool = True,
    ):
        self.split_dir = Path(data_dir) / split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_len_sec * sample_rate)
        self.audio_noise_std = audio_noise_std if is_training else 0.0
        self.is_training = is_training

        self.video_transform = build_video_transform(frame_size, is_training)

        # Discover videos
        self.samples: List[Tuple[str, int]] = []
        for label in [0, 1]:
            class_dir = self.split_dir / str(label)
            if class_dir.exists():
                for vp in sorted(class_dir.glob("*.mp4")):
                    self.samples.append((str(vp), label))

        n0 = sum(1 for _, l in self.samples if l == 0)
        n1 = sum(1 for _, l in self.samples if l == 1)
        print(f"[{split}] {len(self.samples)} videos | class-0: {n0} | class-1: {n1}")

    # ── Frame helpers ──────────────────────────────────────────────────────────

    def _random_frame_indices(self, total: int) -> List[int]:
        """
        Segment-based random sampling: divide video into `num_frames` equal
        segments and draw one frame from each segment. Ensures temporal coverage
        while introducing stochasticity for training augmentation.
        """
        seg = total / self.num_frames
        indices = []
        for i in range(self.num_frames):
            lo = int(i * seg)
            hi = max(lo + 1, int((i + 1) * seg))
            hi = min(hi, total)
            indices.append(random.randint(lo, hi - 1))
        return indices

    def _uniform_frame_indices(self, total: int) -> List[int]:
        """Evenly-spaced frame indices for deterministic eval."""
        return [int(i * total / self.num_frames) for i in range(self.num_frames)]

    def _read_frames(self, video_path: str) -> np.ndarray:
        """Read frames from video; returns uint8 array [T, H, W, 3]."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return np.zeros((self.num_frames, self.frame_size, self.frame_size, 3),
                            dtype=np.uint8)

        indices = (self._random_frame_indices(total) if self.is_training
                   else self._uniform_frame_indices(total))

        frames = []
        last_valid: np.ndarray | None = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_valid = frame
            else:
                frame = (last_valid if last_valid is not None
                         else np.zeros((self.frame_size, self.frame_size, 3),
                                       dtype=np.uint8))
            frames.append(frame)
        cap.release()

        # Safety: pad if short
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else
                          np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))

        return np.stack(frames[:self.num_frames])  # [T, H, W, 3]

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _extract_audio(self, video_path: str) -> torch.Tensor:
        """
        Extract mono 16kHz waveform from video via ffmpeg.
        Returns tensor of shape [T_audio].
        Falls back to silence if extraction fails.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                tmp,
                "-loglevel", "error",
            ], check=True, capture_output=True)

            waveform, sr = torchaudio.load(tmp)          # [1, T]
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            waveform = waveform.squeeze(0)                # [T]

        except Exception as exc:
            print(f"  [WARN] Audio extraction failed for {video_path}: {exc}")
            waveform = torch.zeros(self.sample_rate, dtype=torch.float32)  # 1 s silence
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        # Truncate to max length
        if waveform.shape[0] > self.max_audio_samples:
            waveform = waveform[: self.max_audio_samples]

        return waveform

    def _inject_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Gaussian noise injection for audio augmentation."""
        if self.audio_noise_std > 0:
            noise = torch.randn_like(waveform) * self.audio_noise_std
            return waveform + noise
        return waveform

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        video_path, label = self.samples[idx]

        # ── Video ──────────────────────────────────────────────────────────
        frames_np = self._read_frames(video_path)           # [T, H, W, 3]
        frames = torch.stack([
            self.video_transform(f) for f in frames_np
        ])                                                  # [T, C, H, W]

        # ── Audio ──────────────────────────────────────────────────────────
        waveform = self._extract_audio(video_path)
        waveform = self._inject_noise(waveform)             # augmentation

        return {
            "frames": frames,                               # [T, C, H, W]
            "waveform": waveform,                           # [T_audio]
            "label": torch.tensor(label, dtype=torch.long),
            "video_path": video_path,
        }


# ────────────────────────────────────────────────────────────────────────────────
# Oversampling
# ────────────────────────────────────────────────────────────────────────────────

def get_oversampling_sampler(dataset: BAHVideoDataset) -> WeightedRandomSampler:
    """
    Inverse-frequency weighted sampler to balance A/H (1) vs no-A/H (0).
    Minority class gets up-sampled; effective class ratio → 1:1.
    """
    labels = [lbl for _, lbl in dataset.samples]
    counts = np.bincount(labels)
    class_weights = 1.0 / (counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Collate — pads audio to batch max length
# ────────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    frames = torch.stack([b["frames"] for b in batch])          # [B, T, C, H, W]
    labels = torch.stack([b["label"]  for b in batch])          # [B]
    paths  = [b["video_path"] for b in batch]

    # Pad waveforms to the longest in batch
    waveforms = [b["waveform"] for b in batch]
    max_len   = max(w.shape[0] for w in waveforms)

    padded_waves, attn_masks = [], []
    for w in waveforms:
        pad = max_len - w.shape[0]
        padded_waves.append(torch.nn.functional.pad(w, (0, pad)))
        attn_masks.append(torch.cat([
            torch.ones(w.shape[0],  dtype=torch.long),
            torch.zeros(pad,        dtype=torch.long),
        ]))

    waveforms_t = torch.stack(padded_waves)   # [B, T_audio_max]
    attn_mask_t = torch.stack(attn_masks)     # [B, T_audio_max]

    return {
        "frames":        frames,
        "waveforms":     waveforms_t,
        "attn_masks":    attn_mask_t,
        "labels":        labels,
        "video_paths":   paths,
    }
