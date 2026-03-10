# BAH A/H Recognition — ABAW10 @ CVPR 2026

Binary video-level classification: does a video contain **Ambivalence/Hesitancy**?

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AHVideoClassifier                               │
│                                                                     │
│  Video (32 frames, 224×224)         Audio (raw wav, 16kHz)         │
│       │                                    │                        │
│  Split → 2 clips of 16               HuBERT backbone               │
│  VideoMAE (each clip)                      │                        │
│       │                              [B, T_a, 768]                  │
│  Spatially pool → [B, 8, 768]              │                        │
│  Concat clips  → [B, 16, 768]             ─┤                        │
│       │                             Cross-Modal Transformer         │
│       └──────────────────────────────────→ │                        │
│                                       (2 layers, bidirectional)     │
│                                            │                        │
│                        ┌───────────────────┤                        │
│                        │                   │                        │
│                   Video [B,16,768]    Audio [B,T_a,768]             │
│                        │                   │                        │
│                     BiLSTM           Attn Pool                      │
│                  (2L, hidden=512)          │                        │
│                        │              [B, 768]                      │
│                   mean pool                │                        │
│                   [B, 1024]               ─┤                        │
│                        └──── concat ──────→│                        │
│                                     [B, 1792]                       │
│                                       Linear → GELU → Dropout       │
│                                     [B, 768]                        │
│                                       Linear → GELU → Dropout       │
│                                       Linear → [B, 2]               │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

| Component | Choice | Reason |
|---|---|---|
| Video backbone | VideoMAE-base | Strong spatiotemporal repr from masked autoencoding |
| Audio backbone | HuBERT-base | Excellent self-supervised speech features, captures hesitations, fillers |
| 32 frames | 2×16-frame clips | VideoMAE pretrained on 16 frames; 2 clips doubles temporal coverage |
| Cross-Modal Transformer | 2-layer bidirectional | A/H often manifests as cross-modal conflict — video says yes, audio says no |
| BiLSTM | 2L, hidden=512 | Models temporal dependencies across video clip sequence |
| Oversampling | Inverse-frequency WeightedRandomSampler | Handles A/H class imbalance |
| Audio noise injection | σ=0.005 Gaussian | Augments audio; forces model to learn robust features |
| Differential LR | backbone=5e-6, heads=2e-5 | Preserve pretrained representations while training new layers |
| Weighted CE loss | class-proportional inverse | Additional imbalance correction on top of oversampling |
| Mixed Precision | AMP fp16 | Reduces VRAM, enables larger effective batch size |

---

## Data Structure

```
gs://ah-classify/data/bifurcated/
├── train/
│   ├── 0/   ← no A/H videos (*.mp4)
│   └── 1/   ← A/H present  (*.mp4)
├── val/
│   ├── 0/
│   └── 1/
└── test/
    ├── 0/
    └── 1/
```

---

## Quick Start

### 1. Local test (small data subset)
```bash
pip install -r requirements.txt
python -m trainer.train \
  --gcs_data_path gs://ah-classify/data/bifurcated \
  --num_epochs 5 \
  --batch_size 2
```

### 2. Build & push Docker image
```bash
PROJECT_ID=your-gcp-project-id
IMAGE=gcr.io/$PROJECT_ID/bah-trainer:latest
docker build -t $IMAGE .
docker push $IMAGE
```

### 3. Submit Vertex AI custom job
```bash
export GCP_PROJECT=your-gcp-project-id
export TRAINER_IMAGE=gcr.io/$PROJECT_ID/bah-trainer:latest
python vertex_submit.py
```

### 4. Generate submission predictions (private test set)
```bash
python -m trainer.predict \
  --model_path /tmp/bah_outputs/best_model.pth \
  --video_dir  /path/to/private_test_videos \
  --output_csv predictions.csv
```

---

## Metric

**Primary**: Macro F1 = mean(F1_class0, F1_class1) — both classes equally weighted.

Baseline (Video-LLaVA zero-shot): **P = 0.2827**

---

## Suggested Improvements

1. **Larger backbones**: VideoMAE-large, HuBERT-large for richer representations
2. **MIL (Multiple Instance Learning)**: treat frames as bag; useful since A/H is sparse within video
3. **Temporal contrastive learning**: pre-fine-tune on emotion datasets before BAH
4. **Test-Time Augmentation (TTA)**: average predictions over multiple random samplings
5. **Domain adaptation**: use participant metadata (from meta_data.yml) for personalization
6. **LLM text branch**: add BERT/RoBERTa on Whisper transcripts as a 3rd modality
7. **Ensemble**: train 3–5 models with different seeds and majority vote

---

## File Map

```
bah_ah_recognition/
├── trainer/
│   ├── config.py     — all hyperparameters
│   ├── dataset.py    — BAHVideoDataset, oversampling, collate
│   ├── model.py      — VideoMAE + HuBERT + CrossModalTransformer + BiLSTM
│   ├── metrics.py    — bah_perfs (standalone, competition metrics)
│   ├── train.py      — training loop, GCS integration
│   └── predict.py    — inference + submission CSV generator
├── Dockerfile        — Vertex AI container
├── requirements.txt
├── setup.py
├── vertex_submit.py  — submit Vertex AI custom job
└── README.md
```
