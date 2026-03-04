````markdown
# BAH Multimodal Model (2026 refresh)

Fully rebuilt pipeline for the Ambivalence-Hesitancy (A/H) track at ABAW CVPR 2026. The system follows your latest architectural brief:

- **Video** – ResNet50 backbone (ImageNet weights) run on 32 frames @ 250×250 with random skipping of the final 4–5 convolutional layers and transformer-based temporal aggregation.
- **Audio** – Lightweight Wav2Vec-style encoder that consumes raw 16 kHz audio (10 s max) extracted from each video via ffmpeg.
- **Fusion** – Cross-modal transformer that performs bidirectional audio↔video attention plus a learnable gating layer.
- **Classifier** – Dense neural head used during feature training, followed by a scikit-learn Random Forest for the final binary prediction (0 = absence, 1 = presence of hesitancy).
- **Temporal smoothing** – Configurable smoothing (exponential by default) applied to RF probabilities.
- **Data budget** – Only one tenth of the training split is sampled each epoch to respect the 1/10 requirement.

## Project Layout

```
BAH_model/
├── config/
│   └── config.yaml
├── deploy/
│   ├── build_and_deploy.ps1
│   ├── build_and_deploy.sh
│   └── submit_training_job.py
├── src/
│   ├── constants.py
│   ├── train.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── models/
│       ├── __init__.py
│       ├── audio_encoder.py
│       ├── video_encoder.py
│       ├── fusion.py
│       └── multimodal_model.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Data Sources

Videos live under `gs://ah-classify/data/bifurcated/` with `train/`, `val/`, `test/` folders, each containing `0/` and `1/`. Manifests already exist:

```
gs://ah-classify/data/split/train_0.txt
... train_1.txt
... val_0.txt
... val_1.txt
... test_0.txt
... test_1.txt
```

`src/data/data_loader.py` reads the manifests, downloads each MP4 to a temp directory, samples 32 frames, rescales them to 250×250, and extracts mono 16 kHz audio using ffmpeg. Train manifests are randomly subsampled (10%) on every epoch. Validation/test splits use the full manifest.

## Training Locally

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python -m src.train --config config/config.yaml
```

Artifacts default to `artifacts/<timestamp>/` unless Vertex AI injects `AIP_MODEL_DIR` and `AIP_TENSORBOARD_LOG_DIR`.

## Docker & Vertex AI (Beginner Friendly)

1. **Authenticate**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud auth configure-docker
   ```

2. **Build + push the trainer image**
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/bah-multimodal:latest .
   docker push gcr.io/YOUR_PROJECT_ID/bah-multimodal:latest
   ```

3. **Launch a Vertex AI Custom Job**
   ```bash
   python deploy/submit_training_job.py \
       --project YOUR_PROJECT_ID \
       --region us-central1 \
       --container gcr.io/YOUR_PROJECT_ID/bah-multimodal:latest \
       --config config/config.yaml
   ```
   The script wraps the proper `AIP_MODEL_DIR`, `AIP_TENSORBOARD_LOG_DIR`, machine type, and accelerator (T4) configuration defined under `vertex_ai` in `config.yaml`.

4. **Monitor** via Vertex AI console or Cloud Logging. TensorBoard logs stream to `gs://ah-classify/logs/` (set in config) so you can run `tensorboard --logdir=gs://ah-classify/logs` locally using the GCS fuse or CLI.

## Deployment High-Level

1. Download the contents of `AIP_MODEL_DIR` after training (weights, `random_forest.joblib`, `signature.json`, `config.yaml`).
2. Build an inference image (can reuse Dockerfile) that loads those artifacts and exposes a REST endpoint (FastAPI/Flask) which: extracts features with the encoders, runs the RF head, applies smoothing, and returns logits/probabilities.
3. Deploy on Cloud Run, Vertex Endpoint, or any container platform.

## Evaluation

- `src/evaluation/metrics.py` contains the exact macro-F1 / confusion-matrix code you supplied.
- `src/train.py` prints validation RF metrics immediately after fitting and test metrics (raw + smoothed) before saving artifacts.

## Configuration Cheatsheet (`config/config.yaml`)

- `data.*` – bucket paths, manifest overrides, `data_fraction` (default 0.1).
- `model.video.*` – transformer depth, frame count, skip strategy.
- `model.audio.*` – Wav2Vec hidden dims, layer counts.
- `model.fusion.*` – cross-modal transformer width/depth.
- `model.classifier.*` – Random Forest hyperparameters.
- `model.smoothing.*` – method (`exponential`, `moving_average`, `gaussian`, `median`), window size, alpha, sigma.
- `training.*` – epochs, lr, weight decay, callbacks.
- `vertex_ai.*` – project, region, machine, accelerator, container.

## Tips if You Are New to Docker/GCP

- **Docker**: use `docker run --rm -it <image> bash` to open a shell and ensure ffmpeg/opencv libs work inside the container before shipping to Vertex AI.
- **FFmpeg**: already installed in the Dockerfile to support audio extraction; verify locally with `ffmpeg -version` inside the container.
- **GCS paths**: keep everything in `gs://`—Vertex AI streams data through the bucket, so the training container only downloads what the manifest references.
- **Budget-friendly**: tweak `data.data_fraction` or `training.epochs` as needed; the loader re-samples the 10% subset on every epoch to give broader coverage over time.

## Next Steps

- Implement the inference server (FastAPI + Uvicorn) on top of the saved artifacts.
- Add automated tests for sampling logic and feature extraction.
- Plug the Docker build/push flow into Cloud Build or GitHub Actions for reproducibility.

Need help with any of these follow-up items? Let me know!
````
