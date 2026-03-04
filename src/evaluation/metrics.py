"""Official evaluation metrics for the BAH Ambivalence-Hesitancy challenge."""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import auc, confusion_matrix, f1_score

MACRO_F1 = "MACRO_F1"
W_F1 = "W_F1"
CL_ACC = "CL_ACC"
CFUSE_MARIX = "CONFUSION_MATRIX"
F1_POS = "F1_POS"
F1_NEG = "F1_NEG"
AP_POS = "Average_precision_POS"

__all__ = [
    "MACRO_F1",
    "W_F1",
    "CL_ACC",
    "CFUSE_MARIX",
    "F1_POS",
    "F1_NEG",
    "AP_POS",
    "bah_perfs",
    "set_seed",
]


def softmax(x: np.ndarray, h: float = 1.0) -> np.ndarray:
    e_x = np.exp(x * h)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def average_precision(probs_cl_1: np.ndarray, gt: np.ndarray) -> float:
    conf_step = 0.001
    confidences = np.arange(0, 1, conf_step).tolist()

    n_pos = gt.sum().item()
    assert n_pos > 0, f"{n_pos}"

    l_prec = []
    l_rec = []

    for th in confidences:
        preds = (probs_cl_1 >= th).astype(float)

        tp = np.sum(np.logical_and(preds == 1, gt == 1))
        fp = np.sum(np.logical_and(preds == 1, gt == 0))
        fn = np.sum(np.logical_and(preds == 0, gt == 1))

        assert (tp + fn) == n_pos, f"{tp} | {fn} | {n_pos}"

        prec = 0.0
        if (tp + fp) > 0:
            prec = (tp / (tp + fp)).item()

        rec = (tp / (tp + fn)).item()

        l_prec.append(prec)
        l_rec.append(rec)

    ap = float(auc(np.array(l_rec), np.array(l_prec)))
    return ap


def compute_cnf_mtx(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    conf_mtx = confusion_matrix(
        y_true=target,
        y_pred=pred,
        sample_weight=None,
        normalize="true",
    )
    return conf_mtx


def bah_perfs(gt: np.ndarray, hard_preds: np.ndarray, logits: Optional[np.ndarray] = None) -> Dict[str, float]:
    gt = np.asarray(gt).astype(float)
    hard_preds = np.asarray(hard_preds).astype(float)
    assert gt.shape == hard_preds.shape, f"{gt.shape} | {hard_preds.shape}"

    n = gt.shape[0]
    if logits is not None:
        logits = np.asarray(logits)
        assert logits.ndim == 2, logits.ndim
        assert logits.shape[0] == n, f"{logits.shape[0]} | {n}"

    cl_acc = float((gt == hard_preds).mean().item())
    conf_mtx = compute_cnf_mtx(pred=hard_preds, target=gt)

    f1_s = f1_score(gt.tolist(), hard_preds.tolist(), average=None)
    assert f1_s.shape == (2,), f1_s.shape
    f1_cl_0 = float(f1_s[0].item())
    f1_cl_1 = float(f1_s[1].item())

    macro_f1 = float(np.mean(f1_s).item())
    wf1 = float(f1_score(gt.tolist(), hard_preds.tolist(), average="weighted"))

    ap_cl_1 = 0.0
    if logits is not None:
        probs = softmax(logits, h=1.0)
        ap_cl_1 = average_precision(probs_cl_1=probs[:, 1], gt=gt)

    perfs = {
        CL_ACC: cl_acc,
        CFUSE_MARIX: conf_mtx,
        F1_POS: f1_cl_1,
        F1_NEG: f1_cl_0,
        W_F1: wf1,
        MACRO_F1: macro_f1,
        AP_POS: ap_cl_1,
    }
    return perfs


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _demo() -> None:
    set_seed(0)
    n = 1024
    ncls = 2
    pred_logits = np.random.rand(n, ncls)
    pred = np.argmax(pred_logits, axis=1, keepdims=False)
    gt = np.random.randint(low=0, high=2, size=(n,))

    perfs = bah_perfs(gt, pred, pred_logits)
    for key, value in perfs.items():
        print(f"{key}: {value}")

    perfs = bah_perfs(gt, pred, logits=None)
    for key, value in perfs.items():
        print(f"no-logits {key}: {value}")


if __name__ == "__main__":
    _demo()
