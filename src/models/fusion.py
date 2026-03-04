"""Cross-modal transformer fusion block for video and audio."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .. import constants


class CrossModalFusionBlock(layers.Layer):
    """Single cross-modal fusion block with bidirectional cross-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.video_norm = layers.LayerNormalization(epsilon=1e-6)
        self.audio_norm = layers.LayerNormalization(epsilon=1e-6)
        self.video_cross = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, hidden_dim // num_heads),
            dropout=dropout,
        )
        self.audio_cross = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, hidden_dim // num_heads),
            dropout=dropout,
        )
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential(
            [
                layers.Dense(hidden_dim * 4, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(hidden_dim),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x_video: tf.Tensor, x_audio: tf.Tensor, training: bool = False):
        video_norm = self.video_norm(x_video)
        audio_norm = self.audio_norm(x_audio)

        video_attn = self.video_cross(video_norm, audio_norm, training=training)
        audio_attn = self.audio_cross(audio_norm, video_norm, training=training)
        x_video = x_video + video_attn
        x_audio = x_audio + audio_attn

        fused_tokens = tf.concat([x_video, x_audio], axis=1)
        fused_tokens = fused_tokens + self.ffn(self.ffn_norm(fused_tokens), training=training)
        x_video, x_audio = tf.split(fused_tokens, 2, axis=1)
        return x_video, x_audio


class CrossModalTransformerFusion(keras.Model):
    """Fuses modality embeddings with bidirectional cross-attention."""

    def __init__(
        self,
        hidden_dim: int = constants.FUSION_HIDDEN_DIM,
        num_heads: int = constants.FUSION_NUM_HEADS,
        num_layers: int = constants.FUSION_NUM_LAYERS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._fusion_blocks = [
            CrossModalFusionBlock(hidden_dim, num_heads, dropout, name=f"fusion_block_{i}")
            for i in range(num_layers)
        ]
        self.gate = layers.Dense(hidden_dim, activation="sigmoid")
        self.output_proj = layers.Dense(hidden_dim)
        self._fusion_dropout = layers.Dropout(dropout)

    def call(self, video_feats: tf.Tensor, audio_feats: tf.Tensor, training: bool = False) -> tf.Tensor:
        x_video = tf.expand_dims(video_feats, axis=1)
        x_audio = tf.expand_dims(audio_feats, axis=1)

        for block in self._fusion_blocks:
            x_video, x_audio = block(x_video, x_audio, training=training)

        x_video = tf.squeeze(x_video, axis=1)
        x_audio = tf.squeeze(x_audio, axis=1)
        gate = self.gate(tf.concat([x_video, x_audio], axis=-1))
        fused = gate * x_video + (1.0 - gate) * x_audio
        fused = self._fusion_dropout(fused, training=training)
        return self.output_proj(fused)
