# BAH Multimodal Model - Vertex AI Training Container
# TensorFlow 2.15 with GPU support

FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=3

# Create directories for model artifacts
RUN mkdir -p /app/checkpoints /app/logs

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.train"]

# Default command (can be overridden)
CMD ["--config", "config/config.yaml"]
