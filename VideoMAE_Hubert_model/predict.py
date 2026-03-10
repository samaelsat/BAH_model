"""
BAH A/H Recognition - Inference / Submission Generator
ABAW10 @ CVPR 2026 Challenge

Generates a per-video prediction file for submission.

Usage:
    python -m trainer.predict \
        --model_path /tmp/bah_outputs/best_model.pth \
        --video_dir  /path/to/private_test_videos \
        --output_csv predictions.csv
"""

import os
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchaudio
import cv2
import subprocess
import tempfile

from trainer.model import AHVideoClassifier
from trainer.config import Config
from trainer.metrics import bah_perfs, print_perfs


# ────────────────────────────────────────────────────────────────────────────────
# Lightweight inference dataset (no labels needed)
# ────────────────────────────────────────────────────────────────────────────────

class InferenceDataset(Dataset):
    """
    Expects a directory of .mp4 files (flat or in class sub-folders).
    No labels required — for generating submission predictions.
    """

    def __init__(
        self,
        video_dir: str,
        num_frames: int = 32,
        frame_size: int = 224,
        sample_rate: int = 16000,
        max_audio_len_sec: float = 90.0,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_len_sec * sample_rate)

        self.video_paths = sorted(Path(video_dir).rglob("*.mp4"))
        print(f"InferenceDataset: {len(self.video_paths)} videos in {video_dir}")

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((frame_size, frame_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _uniform_indices(self, total: int):
        return [int(i * total / self.num_frames) for i in range(self.num_frames)]

    def _read_frames(self, path: str):
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
        indices = self._uniform_indices(total)
        frames, last = [], None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last = frame
            else:
                frame = last if last is not None else np.zeros(
                    (self.frame_size, self.frame_size, 3), dtype=np.uint8)
            frames.append(self.transform(frame))
        cap.release()
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        return torch.stack(frames[:self.num_frames])

    def _read_audio(self, path: str):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate), "-ac", "1",
                tmp, "-loglevel", "error",
            ], check=True, capture_output=True)
            wav, sr = torchaudio.load(tmp)
            if sr != self.sample_rate:
                wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
            wav = wav.squeeze(0)
        except Exception:
            wav = torch.zeros(self.sample_rate)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if wav.shape[0] > self.max_audio_samples:
            wav = wav[:self.max_audio_samples]
        return wav

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = str(self.video_paths[idx])
        return {
            "frames":     self._read_frames(path),
            "waveform":   self._read_audio(path),
            "video_path": path,
        }


def infer_collate(batch):
    frames   = torch.stack([b["frames"]   for b in batch])
    waveforms = [b["waveform"] for b in batch]
    max_len  = max(w.shape[0] for w in waveforms)
    padded, masks = [], []
    for w in waveforms:
        p = torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
        m = torch.cat([torch.ones(w.shape[0]), torch.zeros(max_len - w.shape[0])]).long()
        padded.append(p)
        masks.append(m)
    return {
        "frames":      frames,
        "waveforms":   torch.stack(padded),
        "attn_masks":  torch.stack(masks),
        "video_paths": [b["video_path"] for b in batch],
    }


# ────────────────────────────────────────────────────────────────────────────────
# Predict
# ────────────────────────────────────────────────────────────────────────────────

def predict(
    model_path: str,
    video_dir: str,
    output_csv: str,
    batch_size: int = 4,
    num_workers: int = 4,
    gt_labels: dict | None = None,    # optional {video_name: label} for eval
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt   = torch.load(model_path, map_location=device)
    cfg    = ckpt.get("config", Config())
    model  = AHVideoClassifier(
        video_model_name    = cfg.model.video_model_name,
        audio_model_name    = cfg.model.audio_model_name,
        d_model             = cfg.model.d_model,
        nhead               = cfg.model.nhead,
        num_cross_layers    = cfg.model.num_cross_layers,
        dropout             = cfg.model.dropout,
        lstm_hidden         = cfg.model.lstm_hidden,
        lstm_layers         = cfg.model.lstm_layers,
        lstm_bidirectional  = cfg.model.lstm_bidirectional,
        num_classes         = cfg.model.num_classes,
        num_frames_per_clip = cfg.data.num_frames_per_clip,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded from {model_path}")

    ds     = InferenceDataset(video_dir,
                              num_frames=cfg.data.num_frames,
                              frame_size=cfg.data.frame_size,
                              sample_rate=cfg.data.sample_rate)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=infer_collate)

    all_paths, all_logits, all_preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            frames     = batch["frames"].to(device)
            waveforms  = batch["waveforms"].to(device)
            attn_masks = batch["attn_masks"].to(device)

            logits = model(frames, waveforms, attn_masks)
            preds  = torch.argmax(logits, dim=1)

            all_paths.extend(batch["video_paths"])
            all_logits.extend(logits.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "prediction", "logit_0", "logit_1"])
        for path, pred, logit in zip(all_paths, all_preds, all_logits):
            writer.writerow([path, pred, logit[0], logit[1]])

    print(f"Predictions saved to {output_csv}")

    # Optional evaluation (if ground-truth labels provided)
    if gt_labels:
        gt_arr   = np.array([gt_labels[Path(p).name] for p in all_paths])
        pred_arr = np.array(all_preds)
        logit_arr = np.array(all_logits)
        perfs = bah_perfs(gt_arr, pred_arr, logit_arr)
        print_perfs(perfs, prefix="Inference")

    return all_preds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  required=True)
    p.add_argument("--video_dir",   required=True)
    p.add_argument("--output_csv",  default="predictions.csv")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    predict(args.model_path, args.video_dir, args.output_csv,
            args.batch_size, args.num_workers)
