"""
BAH A/H Recognition - Training Script
ABAW10 @ CVPR 2026 Challenge

Usage (local):
    python -m trainer.train

Usage (Vertex AI custom job):
    Submitted via vertex_submit.py — entry point is this module.
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from trainer.config import Config, ModelConfig, DataConfig, TrainConfig
from trainer.dataset import BAHVideoDataset, get_oversampling_sampler, collate_fn
from trainer.model import AHVideoClassifier
from trainer.metrics import bah_perfs, print_perfs, MACRO_F1


# ────────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────────────────────────
# GCS utilities
# ────────────────────────────────────────────────────────────────────────────────

def download_gcs_dir(gcs_uri: str, local_dir: str):
    """Recursively download a GCS directory to local disk."""
    from google.cloud import storage

    uri = gcs_uri.replace("gs://", "")
    bucket_name, prefix = uri.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs  = list(bucket.list_blobs(prefix=prefix))

    print(f"Downloading {len(blobs)} objects from {gcs_uri} → {local_dir}")
    for blob in blobs:
        rel = blob.name[len(prefix):].lstrip("/")
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not blob.name.endswith("/"):
            blob.download_to_filename(dst)
    print("Download complete.")


def upload_file_to_gcs(local_path: str, gcs_uri: str):
    """Upload a single local file to GCS."""
    from google.cloud import storage

    uri = gcs_uri.replace("gs://", "")
    bucket_name, blob_name = uri.split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → {gcs_uri}")


# ────────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_preds, all_labels = [], [], []

    for batch in loader:
        frames     = batch["frames"].to(device)
        waveforms  = batch["waveforms"].to(device)
        attn_masks = batch["attn_masks"].to(device)
        labels     = batch["labels"]

        logits = model(frames, waveforms, attn_masks)
        preds  = torch.argmax(logits, dim=1)

        all_logits.append(logits.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    return bah_perfs(
        gt         = np.concatenate(all_labels),
        hard_preds = np.concatenate(all_preds),
        logits     = np.concatenate(all_logits),
    )


# ────────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # ── Data: download from GCS if not already local ─────────────────────────
    local_data = cfg.data.local_data_dir
    if not os.path.exists(local_data) or not any(Path(local_data).iterdir()):
        os.makedirs(local_data, exist_ok=True)
        download_gcs_dir(cfg.data.gcs_data_path, local_data)

    # ── Datasets ──────────────────────────────────────────────────────────────
    def make_ds(split, training):
        return BAHVideoDataset(
            data_dir          = local_data,
            split             = split,
            num_frames        = cfg.data.num_frames,
            frame_size        = cfg.data.frame_size,
            sample_rate       = cfg.data.sample_rate,
            max_audio_len_sec = cfg.data.max_audio_len_sec,
            audio_noise_std   = cfg.data.audio_noise_std if training else 0.0,
            is_training       = training,
        )

    train_ds = make_ds("train", training=True)
    val_ds   = make_ds("val",   training=False)
    test_ds  = make_ds("test",  training=False)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_sampler = get_oversampling_sampler(train_ds)   # class-balanced oversampling

    loader_kw = dict(
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              sampler=train_sampler, **loader_kw)
    val_loader   = DataLoader(val_ds,  batch_size=cfg.train.batch_size,
                              shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                              shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AHVideoClassifier(
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
        freeze_video_backbone = cfg.model.freeze_video_backbone,
        freeze_audio_backbone = cfg.model.freeze_audio_backbone,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer: differential LRs ──────────────────────────────────────────
    backbone_ids = {
        id(p) for p in list(model.video_backbone.parameters())
                      + list(model.audio_backbone.parameters())
    }
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg.train.backbone_lr},
            {"params": head_params,     "lr": cfg.train.lr},
        ],
        weight_decay=cfg.train.weight_decay,
    )

    total_steps = len(train_loader) * cfg.train.num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Loss: weighted CE to handle residual imbalance ────────────────────────
    labels_list  = [lbl for _, lbl in train_ds.samples]
    counts       = np.bincount(labels_list)
    class_w      = torch.tensor(len(labels_list) / (2.0 * counts),
                                dtype=torch.float).to(device)
    criterion    = nn.CrossEntropyLoss(weight=class_w)

    # ── AMP scaler ────────────────────────────────────────────────────────────
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    scaler  = GradScaler() if use_amp else None

    best_f1    = 0.0
    best_path  = os.path.join(cfg.train.output_dir, "best_model.pth")

    # ════════════════════════════════════════════════════════════════════════════
    # Epoch loop
    # ════════════════════════════════════════════════════════════════════════════
    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for step, batch in enumerate(train_loader):
            frames     = batch["frames"].to(device)
            waveforms  = batch["waveforms"].to(device)
            attn_masks = batch["attn_masks"].to(device)
            labels     = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    logits = model(frames, waveforms, attn_masks)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(frames, waveforms, attn_masks)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            n_batches  += 1

            if step % 20 == 0:
                print(f"  Ep {epoch+1:02d}/{cfg.train.num_epochs} "
                      f"step {step:04d}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch+1} | avg_loss={avg_loss:.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.train.eval_every_n_epochs == 0:
            val_perfs = evaluate(model, val_loader, device)
            print_perfs(val_perfs, prefix=f"Val ep{epoch+1}")

            macro_f1 = val_perfs[MACRO_F1]
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save({
                    "epoch":           epoch + 1,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_macro_f1":   best_f1,
                    "config":          cfg,
                }, best_path)
                print(f"  ✓ New best saved (Macro F1={best_f1:.4f})")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % cfg.train.save_every_n_epochs == 0:
            ckpt = os.path.join(cfg.train.output_dir,
                                f"ckpt_ep{epoch+1:03d}.pth")
            torch.save({"epoch": epoch + 1,
                        "model_state": model.state_dict()}, ckpt)

    # ════════════════════════════════════════════════════════════════════════════
    # Final test evaluation with best model
    # ════════════════════════════════════════════════════════════════════════════
    print("\n── Loading best model for final test evaluation ──")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_perfs = evaluate(model, test_loader, device)
    print_perfs(test_perfs, prefix="TEST")

    # ── Upload best model to GCS ──────────────────────────────────────────────
    try:
        upload_file_to_gcs(
            best_path,
            f"{cfg.train.gcs_output_path}/best_model.pth",
        )
    except Exception as exc:
        print(f"[WARN] GCS upload failed: {exc}")

    return test_perfs


# ────────────────────────────────────────────────────────────────────────────────
# CLI entry point (Vertex AI calls python -m trainer.train)
# ────────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gcs_data_path",    default="gs://ah-classify/data/bifurcated")
    p.add_argument("--gcs_output_path",  default="gs://ah-classify/outputs")
    p.add_argument("--local_data_dir",   default="/tmp/bah_data")
    p.add_argument("--output_dir",       default="/tmp/bah_outputs")
    p.add_argument("--num_epochs",       type=int,   default=30)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--lr",               type=float, default=2e-5)
    p.add_argument("--backbone_lr",      type=float, default=5e-6)
    p.add_argument("--num_frames",       type=int,   default=32)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--freeze_video",     action="store_true")
    p.add_argument("--freeze_audio",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    # Override defaults with CLI args
    cfg.data.gcs_data_path          = args.gcs_data_path
    cfg.data.local_data_dir         = args.local_data_dir
    cfg.data.num_frames             = args.num_frames
    cfg.train.gcs_output_path       = args.gcs_output_path
    cfg.train.output_dir            = args.output_dir
    cfg.train.num_epochs            = args.num_epochs
    cfg.train.batch_size            = args.batch_size
    cfg.train.lr                    = args.lr
    cfg.train.backbone_lr           = args.backbone_lr
    cfg.train.seed                  = args.seed
    cfg.model.freeze_video_backbone = args.freeze_video
    cfg.model.freeze_audio_backbone = args.freeze_audio

    train(cfg)
