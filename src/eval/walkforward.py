"""
Walk-forward evaluation of the HMM regime model.

Method:
- Temporal walk-forward: never use future data for training
- For each fold: train on [t-train_size:t], test on [t:t+test_size]
- Step forward by step_size bars
- Compare against EMA-slope baseline

Usage:
    python -m src.eval.walkforward --symbol BTC/USDT --tf 1h --states auto
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import (
    DEFAULT_N_STATES,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    WF_STEP,
    WF_TEST_SIZE,
    WF_TRAIN_SIZE,
    P_MIN,
)
from src.data.io import get_data, ensure_min_rows
from src.eval.metrics import (
    baseline_ema_sign,
    balanced_direction_accuracy,
    direction_accuracy,
    direction_logloss,
    regime_stability,
    summarize_metrics,
)
from src.features.build_features import build_features, get_feature_matrix
from src.models.hmm_predict import compute_state_stats, decode_states
from src.models.hmm_train import compute_bic_aic, train_hmm

app = typer.Typer()
console = Console()


def run_walkforward(
    df: pd.DataFrame,
    n_states: int,
    train_size: int = WF_TRAIN_SIZE,
    test_size: int = WF_TEST_SIZE,
    step: int = WF_STEP,
    p_min: float = P_MIN,
    auto_states: bool = False,
) -> List[dict]:
    """
    Walk-forward evaluation.

    Args:
        df: OHLCV DataFrame
        n_states: Number of HMM states (or auto-selected)
        train_size, test_size, step: Walk-forward parameters
        p_min: Confidence threshold for signals

    Returns:
        List of per-fold result dicts
    """
    results = []
    n = len(df)
    min_required = train_size + test_size

    if n < min_required:
        raise ValueError(
            f"Not enough data for walk-forward: need {min_required}, got {n}"
        )

    fold = 0
    start = 0

    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        logger.info(
            f"Fold {fold+1}: train [{train_df.index[0].date()} → {train_df.index[-1].date()}] "
            f"| test [{test_df.index[0].date()} → {test_df.index[-1].date()}]"
        )

        try:
            # Build features separately for train / test
            train_feat = build_features(train_df, scale=True)
            X_train = get_feature_matrix(train_feat)

            test_feat = build_features(test_df, scale=True)
            X_test = get_feature_matrix(test_feat)

            if len(X_train) < 200 or len(X_test) < 20:
                logger.warning(f"Fold {fold+1}: insufficient data after feature build, skipping")
                start += step
                fold += 1
                continue

            if auto_states:
                from src.models.model_selection import auto_select_states
                model, selected_n, _ = auto_select_states(X_train, cov_type="diag")
                n_fold = selected_n
            else:
                model, _ = train_hmm(X_train, n_states=n_states)
                n_fold = n_states

            # Decode TEST states (using model trained on train data)
            state_seq_test, posteriors_test = decode_states(model, X_test)
            state_stats = compute_state_stats(model, X_train, *decode_states(model, X_train)[:1])

            # Predicted return for each test bar = mean return of its assigned state
            pred_returns = np.array([
                state_stats.loc[s, "mean_return"] if s in state_stats.index else 0.0
                for s in state_seq_test
            ])

            # True next-bar returns from test data
            true_returns = test_feat["log_return"].values

            # Confidence-filtered predictions
            max_confidence = posteriors_test.max(axis=1)
            confident_mask = max_confidence >= p_min

            # Proba_up per bar
            proba_up_per_bar = np.array([
                state_stats.loc[s, "pct_up"] if s in state_stats.index else 0.5
                for s in state_seq_test
            ])

            # Metrics on all test bars
            dir_acc = direction_accuracy(true_returns, pred_returns)
            bal_acc = balanced_direction_accuracy(true_returns, pred_returns)
            ll = direction_logloss(true_returns, proba_up_per_bar)

            # Baseline: EMA sign
            baseline_pred = baseline_ema_sign(test_df["close"])
            baseline_pred = baseline_pred[:len(true_returns)]
            baseline_acc = direction_accuracy(true_returns, baseline_pred)

            # Regime stability
            state_seq_train, _ = decode_states(model, X_train)
            stability = regime_stability(state_seq_train)

            bic, aic = compute_bic_aic(model, X_train)

            results.append({
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_states": n_fold,
                "dir_accuracy": dir_acc,
                "bal_accuracy": bal_acc,
                "logloss": ll,
                "baseline_acc": baseline_acc,
                "confident_pct": float(confident_mask.mean()),
                "mean_duration": stability.get("mean_duration", np.nan),
                "switch_rate": stability.get("switch_rate", np.nan),
                "bic": bic,
                "aic": aic,
            })

        except Exception as e:
            logger.error(f"Fold {fold+1} error: {e}")

        start += step
        fold += 1

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
@app.command()
def main(
    symbol: str = typer.Option(DEFAULT_SYMBOL),
    tf: str = typer.Option(DEFAULT_TIMEFRAME),
    states: int = typer.Option(DEFAULT_N_STATES),
    auto: bool = typer.Option(False, "--auto", help="Auto-select n_states"),
    since: str = typer.Option("2020-01-01"),
    train_size: int = typer.Option(WF_TRAIN_SIZE),
    test_size: int = typer.Option(WF_TEST_SIZE),
    step: int = typer.Option(WF_STEP),
):
    """Run walk-forward evaluation and print metrics table."""
    df = get_data(symbol=symbol, timeframe=tf, since=since)
    df = ensure_min_rows(df, train_size + test_size)

    results = run_walkforward(
        df,
        n_states=states,
        train_size=train_size,
        test_size=test_size,
        step=step,
        auto_states=auto,
    )

    if not results:
        typer.echo("No walk-forward results computed.")
        raise typer.Exit(1)

    summary = summarize_metrics(results)

    # Rich table
    table = Table(title=f"Walk-Forward Evaluation: {symbol} {tf}", show_lines=True)
    for col in summary.columns:
        table.add_column(col, justify="right")
    for _, row in summary.iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)

    # Aggregate
    console.print(f"\n[bold]Aggregate:[/bold]")
    console.print(f"  Mean Dir. Accuracy : {summary['dir_accuracy'].mean():.4f}")
    console.print(f"  Mean Bal. Accuracy : {summary['bal_accuracy'].mean():.4f}")
    console.print(f"  Mean Log Loss      : {summary['logloss'].mean():.4f}")
    console.print(f"  Mean Baseline Acc  : {summary['baseline_acc'].mean():.4f}")
    console.print(f"  HMM vs Baseline    : {summary['hmm_vs_baseline'].mean():+.4f}")


if __name__ == "__main__":
    app()
