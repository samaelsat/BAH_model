"""Data loader that reads bifurcated videos from GCS and extracts audio."""

from __future__ import annotations

import os
import posixpath
import random
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Tuple

import numpy as np
import tensorflow as tf

from .. import constants


@dataclass
class DataConfig:
    bucket_path: str = constants.GCS_BUCKET
    split_paths: Dict[str, Dict[int, str]] = field(
        default_factory=lambda: {split: paths.copy() for split, paths in constants.GCS_SPLIT_PATHS.items()}
    )
    num_frames: int = constants.NUM_FRAMES
    frame_size: int = constants.VIDEO_FRAME_SIZE
    sample_rate: int = constants.AUDIO_SAMPLE_RATE
    max_duration: int = constants.AUDIO_MAX_DURATION
    batch_size: int = constants.BATCH_SIZE
    data_fraction: float = 0.1
    prefetch_buffer: int = tf.data.AUTOTUNE


class VideoProcessor:
    def __init__(self, num_frames: int, frame_size: int):
        self.num_frames = num_frames
        self.frame_size = frame_size

    def load(self, gcs_path: str) -> Tuple[tf.Tensor, str]:
        temp_dir = tempfile.mkdtemp()
        filename = os.path.basename(gcs_path)
        local_path = os.path.join(temp_dir, filename)
        tf.io.gfile.copy(gcs_path, local_path, overwrite=True)
        frames = self._extract_frames(local_path)
        return frames, local_path

    def _extract_frames(self, video_path: str) -> tf.Tensor:
        import cv2  # Lazy import to keep startup fast

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.num_frames
        indices = np.linspace(0, total - 1, self.num_frames).astype(int)
        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frames.append(frame.astype("float32") / 255.0)
        cap.release()
        return tf.convert_to_tensor(np.stack(frames, axis=0), dtype=tf.float32)


class AudioProcessor:
    def __init__(self, sample_rate: int, max_duration: int):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = sample_rate * max_duration

    def extract(self, video_path: str) -> tf.Tensor:
        wav_path = f"{video_path}_audio.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            wav_path,
        ]
        import subprocess

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            audio_bytes = tf.io.read_file(wav_path)
            audio, _ = tf.audio.decode_wav(audio_bytes, desired_channels=1)
            tf.io.gfile.remove(wav_path)
            audio = tf.squeeze(audio, axis=-1)
            return self._pad_or_truncate(audio)
        except Exception:
            if tf.io.gfile.exists(wav_path):
                tf.io.gfile.remove(wav_path)
            return tf.zeros((self.max_samples,), dtype=tf.float32)

    def _pad_or_truncate(self, audio: tf.Tensor) -> tf.Tensor:
        length = tf.shape(audio)[0]
        audio = audio[: self.max_samples]
        pad = tf.maximum(0, self.max_samples - length)
        return tf.pad(audio, [[0, pad]])


class GCSDataLoader:
    """Creates tf.data pipelines for the bifurcated dataset."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.video = VideoProcessor(config.num_frames, config.frame_size)
        self.audio = AudioProcessor(config.sample_rate, config.max_duration)

    def _read_manifest(self, path: str) -> List[str]:
        with tf.io.gfile.GFile(path, "r") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        resolved = []
        for line in lines:
            if line.startswith("gs://"):
                resolved.append(line)
            else:
                resolved.append(posixpath.join(self.config.bucket_path, line))
        return resolved

    def _collect_samples(self, split: str) -> List[Tuple[str, int]]:
        manifest_map = self.config.split_paths[split]
        samples: List[Tuple[str, int]] = []
        for label, manifest in manifest_map.items():
            paths = self._read_manifest(manifest)
            for path in paths:
                samples.append((path, label))
        if split == "train" and 0 < self.config.data_fraction < 1.0:
            k = max(1, int(len(samples) * self.config.data_fraction))
            samples = random.sample(samples, k)
        return samples

    def _generator(self, split: str) -> Generator[Tuple[Dict[str, tf.Tensor], tf.Tensor], None, None]:
        samples = self._collect_samples(split)
        for gcs_path, label in samples:
            frames, local_path = self.video.load(gcs_path)
            audio = self.audio.extract(local_path)
            tmp_dir = os.path.dirname(local_path)
            if tf.io.gfile.exists(local_path):
                tf.io.gfile.remove(local_path)
            if tf.io.gfile.exists(tmp_dir):
                tf.io.gfile.rmtree(tmp_dir)
            yield {"video": frames, "audio": audio}, tf.convert_to_tensor(label, dtype=tf.int32)

    def get_dataset(self, split: str, shuffle: bool = True) -> tf.data.Dataset:
        output_signature = (
            {
                "video": tf.TensorSpec(
                    shape=(self.config.num_frames, self.config.frame_size, self.config.frame_size, 3),
                    dtype=tf.float32,
                ),
                "audio": tf.TensorSpec(
                    shape=(self.config.sample_rate * self.config.max_duration,),
                    dtype=tf.float32,
                ),
            },
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: self._generator(split),
            output_signature=output_signature,
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=4 * self.config.batch_size)
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(self.config.prefetch_buffer)
        return dataset
