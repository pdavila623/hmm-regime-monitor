"""
Evaluation metrics for regime model quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss


def direction_accuracy(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    """Fraction of bars where predicted sign matches actual sign."""
    true_dir = np.sign(y_true_ret)
    pred_dir = np.sign(y_pred_ret)
    valid = true_dir != 0
    if valid.sum() == 0:
        return 0.5
    return float((true_dir[valid] == pred_dir[valid]).mean())


def balanced_direction_accuracy(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> float:
    """Balanced accuracy for up/down direction prediction."""
    true_dir = (np.sign(y_true_ret) + 1) // 2  # 0 or 1
    pred_dir = (np.sign(y_pred_ret) + 1) // 2
    valid = np.sign(y_true_ret) != 0
    if valid.sum() < 2:
        return 0.5
    return balanced_accuracy_score(true_dir[valid], pred_dir[valid])


def direction_logloss(
    y_true_ret: np.ndarray, proba_up: np.ndarray
) -> float:
    """Log-loss for up/down direction probability."""
    true_dir = (y_true_ret > 0).astype(int)
    proba_up_clipped = np.clip(proba_up, 1e-7, 1 - 1e-7)
    try:
        return log_loss(true_dir, proba_up_clipped)
    except Exception:
        return np.nan


def regime_stability(state_seq: np.ndarray) -> dict:
    """Compute regime stability metrics."""
    if len(state_seq) == 0:
        return {}
    durations = []
    current = state_seq[0]
    dur = 1
    for s in state_seq[1:]:
        if s == current:
            dur += 1
        else:
            durations.append(dur)
            current = s
            dur = 1
    durations.append(dur)
    return {
        "n_switches": len(durations) - 1,
        "mean_duration": float(np.mean(durations)),
        "median_duration": float(np.median(durations)),
        "switch_rate": float((len(durations) - 1) / max(len(state_seq), 1)),
    }


def baseline_ema_sign(
    close: pd.Series, window: int = 20
) -> np.ndarray:
    """
    Simple baseline: predict next-bar direction = sign(EMA slope).
    Returns array of predicted returns (positive or negative).
    """
    ema = close.ewm(span=window, adjust=False).mean()
    slope = ema.diff()
    return np.sign(slope.values)


def summarize_metrics(results: list) -> pd.DataFrame:
    """Summarize walk-forward results into a table."""
    rows = []
    for r in results:
        rows.append({
            "fold": r.get("fold"),
            "train_size": r.get("train_size"),
            "test_size": r.get("test_size"),
            "dir_accuracy": round(r.get("dir_accuracy", np.nan), 4),
            "bal_accuracy": round(r.get("bal_accuracy", np.nan), 4),
            "logloss": round(r.get("logloss", np.nan), 4),
            "baseline_acc": round(r.get("baseline_acc", np.nan), 4),
            "hmm_vs_baseline": round(
                r.get("dir_accuracy", np.nan) - r.get("baseline_acc", np.nan), 4
            ),
            "mean_duration": round(r.get("mean_duration", np.nan), 1),
        })
    return pd.DataFrame(rows)
