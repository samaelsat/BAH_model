"""Training entry point for the rebuilt BAH multimodal model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras

from . import constants
from src.data import DataConfig, GCSDataLoader
from src.evaluation import bah_perfs
from src.models import BAHMultimodalModel, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with tf.io.gfile.GFile(path, "r") as fh:
        return yaml.safe_load(fh)


def setup_strategy() -> tf.distribute.Strategy:
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.TPUStrategy(tpu)
    except Exception:
        gpus = tf.config.list_logical_devices("GPU")
        if len(gpus) > 1:
            return tf.distribute.MirroredStrategy()
        if len(gpus) == 1:
            return tf.distribute.OneDeviceStrategy(device="/gpu:0")
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")


def create_callbacks(model_dir: str, log_dir: str, config: Dict) -> list:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
        ),
    ]
    early_cfg = config.get("training", {}).get("early_stopping", {})
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=early_cfg.get("monitor", "val_loss"),
            patience=early_cfg.get("patience", 10),
            restore_best_weights=True,
        )
    )
    return callbacks


def build_datasets(config: Dict) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    data_cfg = config.get("data", {})
    loader = GCSDataLoader(
        DataConfig(
            bucket_path=data_cfg.get("bucket_path", constants.GCS_BUCKET),
            split_paths=data_cfg.get("split_paths", constants.GCS_SPLIT_PATHS),
            num_frames=data_cfg.get("num_frames", constants.NUM_FRAMES),
            frame_size=data_cfg.get("frame_size", constants.VIDEO_FRAME_SIZE),
            sample_rate=data_cfg.get("sample_rate", constants.AUDIO_SAMPLE_RATE),
            max_duration=data_cfg.get("max_duration", constants.AUDIO_MAX_DURATION),
            batch_size=data_cfg.get("batch_size", constants.BATCH_SIZE),
            data_fraction=data_cfg.get("data_fraction", 0.1),
        )
    )
    return (
        loader.get_dataset("train", shuffle=True),
        loader.get_dataset("val", shuffle=False),
        loader.get_dataset("test", shuffle=False),
    )


def compile_model(model: BAHMultimodalModel, config: Dict) -> None:
    training_cfg = config.get("training", {})
    lr = float(training_cfg.get("learning_rate", constants.LEARNING_RATE))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def extract_features(model: BAHMultimodalModel, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for batch_inputs, batch_labels in dataset:
        fused = model.extract_features(batch_inputs["video"], batch_inputs["audio"], training=False)
        features.append(fused.numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def evaluate_random_forest(model: BAHMultimodalModel, dataset: tf.data.Dataset) -> Dict:
    feats, labels = extract_features(model, dataset)
    probs, preds = model.predict_with_rf(feats)
    metrics = bah_perfs(labels, preds, probs)
    smoothed_probs, smoothed_preds = model.smoother.smooth_probabilities(probs)
    metrics["SMOOTHED"] = bah_perfs(labels, smoothed_preds, logits=None)
    metrics["SMOOTHED"]["probs"] = smoothed_probs
    return metrics


def save_artifacts(model: BAHMultimodalModel, config: Dict, model_dir: str) -> None:
    model.save_weights(os.path.join(model_dir, "nn_weights"))
    model.save_classifier(os.path.join(model_dir, "random_forest.joblib"))
    model.save_signature(os.path.join(model_dir, "signature.json"))
    with tf.io.gfile.GFile(os.path.join(model_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(config, fh)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    strategy = setup_strategy()

    train_ds, val_ds, test_ds = build_datasets(config)
    with strategy.scope():
        model = create_model(config)
        compile_model(model, config)

    model_dir = os.environ.get("AIP_MODEL_DIR", os.path.join("artifacts", datetime.utcnow().strftime("%Y%m%d-%H%M%S")))
    log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR", os.path.join(model_dir, "logs"))
    callbacks = create_callbacks(model_dir, log_dir, config)

    training_cfg = config.get("training", {})
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_cfg.get("epochs", constants.EPOCHS),
        callbacks=callbacks,
    )

    train_feats, train_labels = extract_features(model, train_ds)
    val_feats, val_labels = extract_features(model, val_ds)
    model.train_random_forest(train_feats, train_labels)
    val_probs, val_preds = model.predict_with_rf(val_feats)
    val_metrics = bah_perfs(val_labels, val_preds, val_probs)
    print("Validation metrics:", json.dumps({k: str(v) for k, v in val_metrics.items()}, indent=2))

    test_metrics = evaluate_random_forest(model, test_ds)
    print("Test metrics:", json.dumps({k: str(v) for k, v in test_metrics.items()}, indent=2))

    save_artifacts(model, config, model_dir)


if __name__ == "__main__":
    main()
