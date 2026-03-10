# ── Base: PyTorch 2.1 + CUDA 12.1 ──────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="BAH AH Recognition ABAW10"

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ────────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source ────────────────────────────────────────────────────────────────
COPY . .

# ── Entry point ────────────────────────────────────────────────────────────────
# Vertex AI calls ENTRYPOINT with any additional args from the job spec
ENTRYPOINT ["python", "-m", "trainer.train"]
