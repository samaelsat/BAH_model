"""TensorFlow implementation of a lightweight Wav2Vec-style encoder."""

from __future__ import annotations

from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .. import constants


class ConvFeatureEncoder(layers.Layer):
    """CNN stack mirroring the Wav2Vec 2.0 feature extractor."""

    def __init__(self, conv_layers: List[tuple] | None = None, dropout: float = 0.0):
        super().__init__()
        if conv_layers is None:
            conv_layers = [
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2),
            ]
        self.blocks = []
        for idx, (filters, kernel, stride) in enumerate(conv_layers):
            self.blocks.append(
                keras.Sequential(
                    [
                        layers.Conv1D(filters, kernel, strides=stride, padding="valid", name=f"conv_{idx}"),
                        layers.LayerNormalization(epsilon=1e-5, name=f"ln_{idx}"),
                        layers.Activation("gelu", name=f"gelu_{idx}"),
                        layers.Dropout(dropout),
                    ]
                )
            )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        for block in self.blocks:
            x = block(x, training=training)
        return x


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads),
            dropout=dropout,
        )
        self.dropout1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
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


class Wav2VecEncoder(keras.Model):
    """Wav2Vec-style encoder for raw waveforms."""

    def __init__(
        self,
        hidden_dim: int = 768,
        encoder_dim: int | None = None,
        output_dim: int = constants.FUSION_HIDDEN_DIM,
        sample_rate: int = constants.AUDIO_SAMPLE_RATE,
        max_duration: int = constants.AUDIO_MAX_DURATION,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_weighted_layer_sum: bool = True,
    ):
        super().__init__()
        if encoder_dim is not None:
            hidden_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = sample_rate * max_duration
        self.use_weighted_layer_sum = use_weighted_layer_sum

        self.feature_extractor = ConvFeatureEncoder(dropout=0.0)
        self.projection = layers.Dense(hidden_dim)
        self.feature_norm = layers.LayerNormalization(epsilon=1e-5)
        self.feature_dropout = layers.Dropout(dropout)

        self.max_seq_len = 500
        self.position_embedding = self.add_weight(
            "audio_pos",
            shape=(1, self.max_seq_len, hidden_dim),
            initializer="zeros",
            trainable=True,
        )

        self.encoder_layers = [
            TransformerEncoderBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self.final_norm = layers.LayerNormalization(epsilon=1e-5)

        if self.use_weighted_layer_sum:
            self.layer_weights = self.add_weight(
                "layer_weights",
                shape=(num_layers + 1,),
                initializer="ones",
                trainable=True,
            )

        self.output_dropout = layers.Dropout(dropout)
        self.output_projection = layers.Dense(output_dim)

    def _pad_or_truncate(self, audio: tf.Tensor) -> tf.Tensor:
        audio = audio[:, : self.max_samples, :]
        length = tf.shape(audio)[1]
        padding = tf.maximum(0, self.max_samples - length)
        audio = tf.pad(audio, [[0, 0], [0, padding], [0, 0]])
        return audio

    def call(self, audio: tf.Tensor, training: bool = False) -> tf.Tensor:
        if audio.shape.rank == 1:
            audio = tf.expand_dims(audio, 0)
        if audio.shape.rank == 2:
            audio = tf.expand_dims(audio, -1)

        audio = self._pad_or_truncate(audio)
        features = self.feature_extractor(audio, training=training)
        seq_len = tf.shape(features)[1]

        hidden = self.projection(features)
        hidden = self.feature_norm(hidden)
        hidden = self.feature_dropout(hidden, training=training)
        hidden = hidden + self.position_embedding[:, :seq_len, :]

        outputs = [hidden]
        x = hidden
        for layer in self.encoder_layers:
            x = layer(x, training=training)
            outputs.append(x)
        x = self.final_norm(x)

        if self.use_weighted_layer_sum:
            weights = tf.nn.softmax(self.layer_weights)
            stacked = tf.stack(outputs, axis=0)
            x = tf.reduce_sum(stacked * weights[:, None, None, None], axis=0)

        x = tf.reduce_mean(x, axis=1)
        x = self.output_dropout(x, training=training)
        return self.output_projection(x)


def create_audio_encoder(**kwargs) -> Wav2VecEncoder:
    return Wav2VecEncoder(**kwargs)
