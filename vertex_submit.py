"""
BAH A/H Recognition - Vertex AI Custom Job Submission
ABAW10 @ CVPR 2026 Challenge

Steps to run:
  1. Build & push Docker image:
       PROJECT_ID=your-gcp-project-id
       IMAGE_URI=gcr.io/$PROJECT_ID/bah-trainer:latest
       docker build -t $IMAGE_URI .
       docker push $IMAGE_URI

  2. Set PROJECT_ID and IMAGE_URI below, then:
       python vertex_submit.py
"""

import os
from google.cloud import aiplatform

# ── Configure these ────────────────────────────────────────────────────────────
PROJECT_ID   = os.environ.get("GCP_PROJECT",  "your-gcp-project-id")
REGION       = os.environ.get("GCP_REGION",   "us-central1")
IMAGE_URI    = os.environ.get("TRAINER_IMAGE", f"gcr.io/{PROJECT_ID}/bah-trainer:latest")
BUCKET       = "ah-classify"
STAGING_URI  = f"gs://{BUCKET}/vertex-staging"
# ──────────────────────────────────────────────────────────────────────────────

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=STAGING_URI,
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="bah-ah-recognition-abaw10",
    container_uri=IMAGE_URI,
    model_serving_container_image_uri=None,  # not deploying as endpoint
)

# Training args forwarded to trainer/train.py
ARGS = [
    "--gcs_data_path",   "gs://ah-classify/data/bifurcated",
    "--gcs_output_path", "gs://ah-classify/outputs",
    "--local_data_dir",  "/tmp/bah_data",
    "--output_dir",      "/tmp/bah_outputs",
    "--num_epochs",      "30",
    "--batch_size",      "4",
    "--lr",              "2e-5",
    "--backbone_lr",     "5e-6",
    "--num_frames",      "32",
    "--seed",            "42",
]

model = job.run(
    args=ARGS,
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=1,
    base_output_dir=f"gs://{BUCKET}/outputs",
    sync=True,    # block until job completes
)

print("Training job complete.")
print(f"Best model saved to: gs://{BUCKET}/outputs/best_model.pth")
