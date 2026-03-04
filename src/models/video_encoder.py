"""Video encoder built on ResNet50 with temporal aggregation."""

from __future__ import annotations

import random
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.applications.resnet50 import preprocess_input

from .. import constants


class TemporalTransformerBlock(layers.Layer):
    """Transformer block for aggregating frame-level embeddings."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads),
            dropout=dropout,
        )
        self.dropout1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_out = self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.dropout1(attn_out, training=training)
        ffn_out = self.ffn(self.norm2(x), training=training)
        return x + ffn_out


_BACKBONE_FACTORIES = {
    "resnet50": ResNet50,
    "resnet101": ResNet101,
}


class ResNetVideoEncoder(keras.Model):
    """Frame-level ResNet50 encoder with transformer-based temporal pooling."""

    def __init__(
        self,
        num_frames: int = constants.NUM_FRAMES,
        frame_size: int = constants.VIDEO_FRAME_SIZE,
        output_dim: int = constants.FUSION_HIDDEN_DIM,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_ff_dim: int = 1024,
        dropout: float = 0.3,
        skip_last_layer_choices: Tuple[int, ...] = (4, 5),
        freeze_backbone: bool = True,
        base_cnn: str = "resnet50",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.output_dim = output_dim
        self.dropout = dropout

        skip_count = random.choice(skip_last_layer_choices)
        builder = _BACKBONE_FACTORIES.get(base_cnn.lower())
        if builder is None:
            supported = ", ".join(sorted(_BACKBONE_FACTORIES))
            raise ValueError(f"Unsupported base_cnn '{base_cnn}'. Supported: {supported}")
        base = builder(include_top=False, weights="imagenet", pooling=None)
        if skip_count > 0:
            trimmed_output = base.layers[-(skip_count + 1)].output
            self.backbone = keras.Model(inputs=base.input, outputs=trimmed_output)
        else:
            self.backbone = base
        self.backbone.trainable = not freeze_backbone

        self.global_pool = layers.GlobalAveragePooling2D()
        self.temporal_pos = self.add_weight(
            "temporal_pos",
            shape=(1, num_frames, self.backbone.output_shape[-1]),
            initializer="zeros",
            trainable=True,
        )
        self.temporal_blocks = [
            TemporalTransformerBlock(
                embed_dim=self.backbone.output_shape[-1],
                num_heads=temporal_heads,
                ff_dim=temporal_ff_dim,
                dropout=dropout,
            )
            for _ in range(temporal_layers)
        ]
        self.temporal_norm = layers.LayerNormalization(epsilon=1e-6)
        self.temporal_dropout = layers.Dropout(dropout)
        self.project = layers.Dense(output_dim)

    def call(self, frames: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch = tf.shape(frames)[0]
        frames = tf.reshape(frames, (batch * self.num_frames, self.frame_size, self.frame_size, 3))
        frames = tf.image.resize(frames, (self.frame_size, self.frame_size))
        frames = preprocess_input(frames * 255.0)
        feats = self.backbone(frames, training=training)
        feats = self.global_pool(feats)
        feats = tf.reshape(feats, (batch, self.num_frames, -1))

        feats += self.temporal_pos[:, : self.num_frames, :]
        for block in self.temporal_blocks:
            feats = block(feats, training=training)

        feats = self.temporal_norm(feats)
        feats = tf.reduce_mean(feats, axis=1)
        feats = self.temporal_dropout(feats, training=training)
        return self.project(feats)


def create_video_encoder(**kwargs) -> ResNetVideoEncoder:
    return ResNetVideoEncoder(**kwargs)
