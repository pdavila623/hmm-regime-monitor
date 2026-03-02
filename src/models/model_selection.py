"""
Automatic model selection: choose best n_states via BIC/AIC.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from src.config import DEFAULT_COV_TYPE, MAX_STATES, MIN_STATES
from src.models.hmm_train import train_hmm, compute_bic_aic


def auto_select_states(
    X: np.ndarray,
    min_states: int = MIN_STATES,
    max_states: int = MAX_STATES,
    cov_type: str = DEFAULT_COV_TYPE,
    criterion: str = "bic",
) -> Tuple[GaussianHMM, int, List[Dict[str, Any]]]:
    """
    Evaluate HMMs with n_states in [min_states, max_states] and return the best.

    Args:
        X: Feature matrix
        min_states, max_states: Range of states to evaluate
        cov_type: Covariance type
        criterion: 'bic' (recommended) or 'aic'

    Returns:
        (best_model, best_n_states, ranking_list)
    """
    ranking = []
    best_model = None
    best_score = np.inf
    best_n = min_states

    for n in range(min_states, max_states + 1):
        logger.info(f"Evaluating {n} states...")
        try:
            model, ll = train_hmm(X, n_states=n, cov_type=cov_type)
            bic, aic = compute_bic_aic(model, X)
            score = bic if criterion == "bic" else aic

            ranking.append({
                "n_states": n,
                "log_likelihood": ll,
                "bic": bic,
                "aic": aic,
                "score": score,
                "model": model,
            })

            if score < best_score:
                best_score = score
                best_model = model
                best_n = n

        except Exception as e:
            logger.warning(f"  n_states={n} failed: {e}")

    # Sort by criterion score
    ranking.sort(key=lambda x: x["score"])
    logger.info(f"Best n_states={best_n} with {criterion.upper()}={best_score:.1f}")
    return best_model, best_n, ranking
