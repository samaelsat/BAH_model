"""
BAH A/H Recognition - Evaluation Metrics
ABAW10 @ CVPR 2026 Challenge

Standalone port of bah_metrics.py (no dependency on external `constants` module).
Primary metric: Macro F1 = average of F1_class0 and F1_class1.
"""

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, auc

# ── Metric key constants ──────────────────────────────────────────────────────
MACRO_F1     = "MACRO_F1"           # main competition metric
W_F1         = "W_F1"
CL_ACC       = "CL_ACC"
CFUSE_MARIX  = "CONFUSION_MATRIX"
F1_POS       = "F1_POS"            # F1 for class 1 (A/H present)
F1_NEG       = "F1_NEG"            # F1 for class 0 (no A/H)
AP_POS       = "Average_precision_POS"


def _softmax(x: np.ndarray, h: float = 1.0) -> np.ndarray:
    shifted = x * h - np.max(x * h, axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=1, keepdims=True)


def _average_precision(probs_pos: np.ndarray, gt: np.ndarray) -> float:
    """Compute AP for the positive class by sweeping confidence thresholds."""
    n_pos = int(gt.sum())
    if n_pos == 0:
        return 0.0

    thresholds = np.arange(0.0, 1.0, 0.001)
    precisions, recalls = [], []

    for th in thresholds:
        preds = (probs_pos >= th).astype(float)
        tp = float(np.logical_and(preds == 1, gt == 1).sum())
        fp = float(np.logical_and(preds == 1, gt == 0).sum())
        fn = float(np.logical_and(preds == 0, gt == 1).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn)
        precisions.append(prec)
        recalls.append(rec)

    rec_arr  = np.array(recalls)
    prec_arr = np.array(precisions)
    return float(auc(rec_arr, prec_arr))


def bah_perfs(
    gt: np.ndarray,
    hard_preds: np.ndarray,
    logits: np.ndarray | None = None,
) -> dict:
    """
    Compute all BAH challenge metrics.

    Args:
        gt          : ground-truth labels (n,)  — 0 or 1
        hard_preds  : predicted labels   (n,)  — 0 or 1
        logits      : raw model logits   (n, 2) — optional, needed for AP

    Returns dict with keys: MACRO_F1, W_F1, CL_ACC, CONFUSION_MATRIX,
                             F1_POS, F1_NEG, Average_precision_POS
    """
    gt         = gt.astype(float)
    hard_preds = hard_preds.astype(float)

    cl_acc   = float((gt == hard_preds).mean())
    conf_mtx = confusion_matrix(gt, hard_preds, normalize="true")

    f1_per_class = f1_score(gt.tolist(), hard_preds.tolist(), average=None,
                            zero_division=0)
    if f1_per_class.shape[0] < 2:
        # Edge case: only one class predicted
        f1_per_class = np.pad(f1_per_class, (0, 2 - f1_per_class.shape[0]))

    macro_f1 = float(np.mean(f1_per_class))
    wf1      = float(f1_score(gt.tolist(), hard_preds.tolist(),
                              average="weighted", zero_division=0))

    ap_pos = 0.0
    if logits is not None:
        probs  = _softmax(logits)
        ap_pos = _average_precision(probs[:, 1], gt)

    return {
        CL_ACC:      cl_acc,
        CFUSE_MARIX: conf_mtx,
        F1_POS:      float(f1_per_class[1]),
        F1_NEG:      float(f1_per_class[0]),
        W_F1:        wf1,
        MACRO_F1:    macro_f1,
        AP_POS:      ap_pos,
    }


def print_perfs(perfs: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}Macro F1 : {perfs[MACRO_F1]:.4f}   ← competition metric")
    print(f"{tag}F1 Pos   : {perfs[F1_POS]:.4f}")
    print(f"{tag}F1 Neg   : {perfs[F1_NEG]:.4f}")
    print(f"{tag}W-F1     : {perfs[W_F1]:.4f}")
    print(f"{tag}Accuracy : {perfs[CL_ACC]:.4f}")
    print(f"{tag}AP (pos) : {perfs[AP_POS]:.4f}")
    print(f"{tag}Confusion matrix (rows=true, cols=pred):")
    print(perfs[CFUSE_MARIX])
