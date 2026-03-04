"""Multimodal model combining video, audio, and cross-modal fusion."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from .. import constants
from src.models.audio_encoder import Wav2VecEncoder
from src.models.fusion import CrossModalTransformerFusion
from src.models.video_encoder import ResNetVideoEncoder


class TemporalSmoothing:
    """Applies temporal smoothing to probability sequences."""

    def __init__(self, method: str = "exponential", window_size: int = 5, alpha: float = 0.3, sigma: float = 1.0):
        self.method = method
        self.window_size = window_size
        self.alpha = alpha
        self.sigma = sigma

    def smooth_probabilities(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if probs.size == 0:
            return probs, np.array([], dtype=int)
        if self.method == "moving_average":
            return self._moving_average(probs)
        if self.method == "gaussian":
            return self._gaussian(probs)
        if self.method == "median":
            return self._median(probs)
        return self._exponential(probs)

    def _moving_average(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kernel = np.ones(self.window_size) / self.window_size
        padded = np.pad(probs, ((self.window_size // 2, self.window_size // 2), (0, 0)), mode="edge")
        smoothed = np.array([
            np.convolve(padded[:, idx], kernel, mode="valid") for idx in range(probs.shape[1])
        ]).T
        return smoothed, np.argmax(smoothed, axis=-1)

    def _gaussian(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        half = self.window_size // 2
        x = np.arange(-half, half + 1)
        kernel = np.exp(-(x**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        padded = np.pad(probs, ((half, half), (0, 0)), mode="edge")
        smoothed = np.array([
            np.convolve(padded[:, idx], kernel, mode="valid") for idx in range(probs.shape[1])
        ]).T
        return smoothed, np.argmax(smoothed, axis=-1)

    def _median(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        half = self.window_size // 2
        padded = np.pad(probs, ((half, half), (0, 0)), mode="edge")
        smoothed = []
        for i in range(probs.shape[0]):
            window = padded[i : i + self.window_size]
            smoothed.append(np.median(window, axis=0))
        smoothed = np.array(smoothed)
        return smoothed, np.argmax(smoothed, axis=-1)

    def _exponential(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if probs.shape[0] == 0:
            return probs, np.array([], dtype=int)
        smoothed = np.zeros_like(probs)
        smoothed[0] = probs[0]
        for idx in range(1, probs.shape[0]):
            smoothed[idx] = self.alpha * probs[idx] + (1.0 - self.alpha) * smoothed[idx - 1]
        return smoothed, np.argmax(smoothed, axis=-1)


class RandomForestHead:
    """Scikit-learn RandomForest wrapper with scaling."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        scaled = self.scaler.fit_transform(features)
        self.classifier.fit(scaled, labels)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._check()
        scaled = self.scaler.transform(features)
        return self.classifier.predict(scaled)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._check()
        scaled = self.scaler.transform(features)
        return self.classifier.predict_proba(scaled)

    def save(self, path: str) -> None:
        joblib.dump({"classifier": self.classifier, "scaler": self.scaler, "is_fitted": self.is_fitted}, path)

    def load(self, path: str) -> None:
        state = joblib.load(path)
        self.classifier = state["classifier"]
        self.scaler = state["scaler"]
        self.is_fitted = state.get("is_fitted", True)

    def _check(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("RandomForestHead is not fitted yet.")


class BAHMultimodalModel(keras.Model):
    """Full multimodal network used for feature extraction."""

    def __init__(
        self,
        video_cfg: Optional[Dict] = None,
        audio_cfg: Optional[Dict] = None,
        fusion_cfg: Optional[Dict] = None,
        classifier_cfg: Optional[Dict] = None,
        smoothing_cfg: Optional[Dict] = None,
        num_classes: int = constants.NUM_CLASSES,
    ):
        super().__init__()
        video_cfg = video_cfg or {}
        audio_cfg = audio_cfg or {}
        fusion_cfg = fusion_cfg or {}
        classifier_cfg = classifier_cfg or {}
        smoothing_cfg = smoothing_cfg or {}

        self.video_encoder = ResNetVideoEncoder(**video_cfg)
        self.audio_encoder = Wav2VecEncoder(**audio_cfg)
        self.fusion = CrossModalTransformerFusion(**fusion_cfg)

        self.nn_head = keras.Sequential(
            [
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(num_classes),
            ]
        )

        self.rf_head = RandomForestHead(**classifier_cfg)
        self.smoother = TemporalSmoothing(**smoothing_cfg)
        self.num_classes = num_classes

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        video = inputs["video"]
        audio = inputs["audio"]
        fused = self.extract_features(video, audio, training=training)
        return self.nn_head(fused, training=training)

    def extract_features(self, video: tf.Tensor, audio: tf.Tensor, training: bool = False) -> tf.Tensor:
        video_feats = self.video_encoder(video, training=training)
        audio_feats = self.audio_encoder(audio, training=training)
        fused = self.fusion(video_feats, audio_feats, training=training)
        return fused

    def train_random_forest(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.rf_head.fit(features, labels)

    def predict_with_rf(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.rf_head.predict_proba(features)
        preds = np.argmax(probs, axis=-1)
        return probs, preds

    def save_classifier(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.rf_head.save(path)

    def load_classifier(self, path: str) -> None:
        self.rf_head.load(path)

    def save_signature(self, path: str) -> None:
        signature = {
            "inputs": {
                "video": {
                    "shape": [None, constants.NUM_FRAMES, constants.VIDEO_FRAME_SIZE, constants.VIDEO_FRAME_SIZE, 3],
                    "dtype": "float32",
                },
                "audio": {
                    "shape": [None, constants.AUDIO_SAMPLE_RATE * constants.AUDIO_MAX_DURATION],
                    "dtype": "float32",
                },
            },
            "outputs": {
                "logits": {
                    "shape": [None, self.num_classes],
                    "dtype": "float32",
                }
            },
        }
        with tf.io.gfile.GFile(path, "w") as fh:
            json.dump(signature, fh, indent=2)


def create_model(config: Dict) -> BAHMultimodalModel:
    model_cfg = config.get("model", {})
    video_cfg = model_cfg.get("video", {})
    audio_cfg = model_cfg.get("audio", {})
    fusion_cfg = model_cfg.get("fusion", {})
    classifier_cfg = config.get("random_forest", model_cfg.get("classifier", {}))
    smoothing_cfg = config.get("smoothing", model_cfg.get("smoothing", {}))

    if "num_frames" not in video_cfg:
        video_cfg["num_frames"] = constants.NUM_FRAMES
    if "frame_size" not in video_cfg:
        video_cfg["frame_size"] = constants.VIDEO_FRAME_SIZE
    if "sample_rate" not in audio_cfg:
        audio_cfg["sample_rate"] = constants.AUDIO_SAMPLE_RATE
    if "max_duration" not in audio_cfg:
        audio_cfg["max_duration"] = constants.AUDIO_MAX_DURATION

    return BAHMultimodalModel(
        video_cfg=video_cfg,
        audio_cfg=audio_cfg,
        fusion_cfg=fusion_cfg,
        classifier_cfg=classifier_cfg,
        smoothing_cfg=smoothing_cfg,
        num_classes=model_cfg.get("num_classes", constants.NUM_CLASSES),
    )
