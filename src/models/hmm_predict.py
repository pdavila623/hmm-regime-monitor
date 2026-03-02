"""
HMM prediction: decode states, compute posteriors, label regimes.

Regime labeling is done POST-HOC based on each state's empirical statistics
(mean return, volatility) on the training data. This avoids look-ahead bias
in the labeling itself.

Signal logic:
- Current state = Viterbi decoded state for last bar
- Confidence = max posterior probability for last bar
- If confidence < p_min → "Neutral / Uncertain"
- Regime label: Bull / Bear / Sideways / Volatile Bull / Volatile Bear
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from src.config import P_MIN


def decode_states(model: GaussianHMM, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Viterbi (state_sequence) and forward-backward (posteriors).

    Returns:
        state_seq: (n_samples,) most probable state sequence
        posteriors: (n_samples, n_states) posterior probabilities
    """
    log_prob, state_seq = model.decode(X, algorithm="viterbi")
    posteriors = model.predict_proba(X)
    return state_seq, posteriors


def compute_state_stats(
    model: GaussianHMM,
    X: np.ndarray,
    state_seq: np.ndarray,
    close_returns: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute per-state statistics for regime labeling and dashboard.

    Args:
        model: Trained GaussianHMM
        X: Feature matrix
        state_seq: Viterbi state sequence
        close_returns: Raw log returns (for directional stats)

    Returns:
        DataFrame with state stats
    """
    n_states = model.n_components
    stats = []

    for s in range(n_states):
        mask = state_seq == s
        count = mask.sum()
        if count == 0:
            stats.append({
                "state": s,
                "count": 0,
                "pct": 0.0,
                "mean_return": 0.0,
                "vol": 0.0,
                "sharpe": 0.0,
                "pct_up": 0.5,
                "avg_duration": 0.0,
                "label": "Unknown",
            })
            continue

        # Use raw returns (feature col 0 is log_return)
        ret = X[mask, 0] if close_returns is None else close_returns[mask]
        mean_ret = ret.mean()
        vol = ret.std() + 1e-10
        sharpe = mean_ret / vol * np.sqrt(252)  # annualized (approx)
        pct_up = (ret > 0).mean()

        # Average state duration
        durations = []
        current_dur = 1
        for i in range(1, len(state_seq)):
            if state_seq[i] == s and state_seq[i - 1] == s:
                current_dur += 1
            elif state_seq[i - 1] == s:
                durations.append(current_dur)
                current_dur = 1
        avg_duration = np.mean(durations) if durations else count

        stats.append({
            "state": s,
            "count": int(count),
            "pct": round(100 * count / len(state_seq), 1),
            "mean_return": round(float(mean_ret), 6),
            "vol": round(float(vol), 6),
            "sharpe": round(float(sharpe), 3),
            "pct_up": round(float(pct_up), 3),
            "avg_duration": round(float(avg_duration), 1),
        })

    df_stats = pd.DataFrame(stats).set_index("state")
    df_stats["label"] = _label_states(df_stats)
    return df_stats


def _label_states(stats: pd.DataFrame) -> pd.Series:
    """
    Assign human-readable labels based on return and vol profile.

    Strategy:
    - Split states by median volatility into High-vol / Low-vol
    - Within each group, label by mean return:
        High-vol + positive return → "Volatile Bull"
        High-vol + negative return → "Volatile Bear"
        Low-vol + positive return → "Bull"
        Low-vol + negative return → "Bear"
        Low-vol + near-zero return → "Sideways"
    """
    labels = {}
    if len(stats) == 0:
        return pd.Series(labels)

    med_vol = stats["vol"].median()
    med_ret = stats["mean_return"].median()
    ret_threshold = stats["mean_return"].std() * 0.3  # near-zero band

    for s, row in stats.iterrows():
        high_vol = row["vol"] >= med_vol
        ret = row["mean_return"]

        if high_vol:
            label = "Volatile Bull" if ret > 0 else "Volatile Bear"
        else:
            if abs(ret) <= ret_threshold:
                label = "Sideways"
            elif ret > 0:
                label = "Bull"
            else:
                label = "Bear"
        labels[s] = label

    # Handle duplicates by appending state number
    seen = {}
    for s, label in labels.items():
        if label in seen:
            labels[s] = f"{label} #{s}"
        seen[label] = s

    return pd.Series(labels)


def get_current_signal(
    model: GaussianHMM,
    X: np.ndarray,
    state_stats: pd.DataFrame,
    p_min: float = P_MIN,
) -> Dict:
    """
    Compute the current regime signal from the last bar.

    Returns dict with:
        state: int state index
        label: regime label
        confidence: max posterior probability
        signal: 'Bull'/'Bear'/'Sideways'/'Neutral'
        expected_return: mean return for current state
        expected_vol: vol for current state
        proba_up: probability of positive next return
    """
    posteriors = model.predict_proba(X)
    last_posteriors = posteriors[-1]
    current_state = int(np.argmax(last_posteriors))
    confidence = float(last_posteriors[current_state])

    if confidence < p_min:
        return {
            "state": current_state,
            "label": "Neutral / Uncertain",
            "signal": "Neutral",
            "confidence": round(confidence, 4),
            "expected_return": 0.0,
            "expected_vol": 0.0,
            "proba_up": 0.5,
            "posteriors": last_posteriors,
        }

    row = state_stats.loc[current_state] if current_state in state_stats.index else None
    label = row["label"] if row is not None else "Unknown"
    signal = label.split(" ")[0] if label not in ("Unknown", "Neutral / Uncertain") else "Neutral"

    return {
        "state": current_state,
        "label": label,
        "signal": signal,
        "confidence": round(confidence, 4),
        "expected_return": float(row["mean_return"]) if row is not None else 0.0,
        "expected_vol": float(row["vol"]) if row is not None else 0.0,
        "proba_up": float(row["pct_up"]) if row is not None else 0.5,
        "posteriors": last_posteriors,
    }
