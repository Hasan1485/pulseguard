#!/usr/bin/env python3
"""PulseGuard model training.

Downloads the ULB credit-card fraud dataset from OpenML (284,807 transactions,
492 frauds), enriches it with per-card rolling behavioral features (the same
features the online Redis feature store computes), balances the training split
with SMOTE, trains an XGBoost classifier, and exports:

  model/fraud_xgb.onnx   ONNX model (ZipMap stripped -> raw probability tensor)
  model/metrics.json     held-out evaluation metrics
  data/stream.csv.gz     time-ordered held-out transactions for live replay

Run:  python training/train.py
"""

import gzip
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

N_CARDS = 2000
WINDOW_SEC = 60.0
TSL_CAP = 3600.0
SEED = 42

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_NAMES = ["Amount", *V_COLS, "txn_count_60s", "time_since_last", "amount_ratio"]


def download() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    print("Downloading/loading 'creditcard' from OpenML (cached after first run)...")
    t0 = time.time()
    ds = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    df = ds.frame
    df["Class"] = df["Class"].astype(int)
    # The OpenML export drops the original 'Time' column but preserves row
    # order (the dataset covers ~2 days of transactions, time-ordered).
    # Reconstruct monotonic event times with seeded exponential inter-arrivals
    # at the dataset's true average rate (~0.6 s/txn).
    rng = np.random.default_rng(SEED)
    df["Time"] = np.cumsum(rng.exponential(172_800 / len(df), size=len(df)))
    print(f"  {len(df):,} rows, {df['Class'].sum():,} frauds ({time.time() - t0:.0f}s)")
    return df


def assign_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a synthetic card id to every transaction (skewed like real usage)."""
    rng = np.random.default_rng(SEED)
    weights = rng.dirichlet(np.full(N_CARDS, 0.5))
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df["card_id"] = rng.choice(N_CARDS, size=len(df), p=weights)
    return df


def rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card rolling aggregates, computed exactly like the online feature
    store: state is read BEFORE the current transaction is added."""
    times: dict[int, deque] = {}
    counts: dict[int, int] = {}
    sums: dict[int, float] = {}
    lasts: dict[int, float] = {}

    n = len(df)
    f_count = np.zeros(n, dtype=np.float32)
    f_tsl = np.zeros(n, dtype=np.float32)
    f_ratio = np.zeros(n, dtype=np.float32)

    for i, (card, ts, amount) in enumerate(
        zip(
            df["card_id"].to_numpy(),
            df["Time"].to_numpy(),
            df["Amount"].to_numpy(),
            strict=True,
        )
    ):
        dq = times.setdefault(card, deque())
        while dq and ts - dq[0] > WINDOW_SEC:
            dq.popleft()
        f_count[i] = len(dq)
        f_tsl[i] = min(ts - lasts[card], TSL_CAP) if card in lasts else TSL_CAP
        c = counts.get(card, 0)
        if c > 0 and sums[card] > 0:
            f_ratio[i] = amount / (sums[card] / c)
        else:
            f_ratio[i] = 1.0
        dq.append(ts)
        counts[card] = c + 1
        sums[card] = sums.get(card, 0.0) + float(amount)
        lasts[card] = ts

    df["txn_count_60s"] = f_count
    df["time_since_last"] = f_tsl
    df["amount_ratio"] = f_ratio
    return df


def train(df: pd.DataFrame):
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import (
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy()

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

    print(f"Applying SMOTE: train {len(y_tr):,} rows, {y_tr.sum():,} frauds ->", end=" ")
    X_bal, y_bal = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    print(f"{len(y_bal):,} rows balanced")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
        random_state=SEED,
    )
    print("Training XGBoost...")
    t0 = time.time()
    model.fit(X_bal, y_bal)
    print(f"  done in {time.time() - t0:.0f}s")

    proba = model.predict_proba(X_te)[:, 1]
    threshold = 0.5
    pred = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "threshold": threshold,
        "test_rows": int(len(y_te)),
        "test_frauds": int(y_te.sum()),
        "features": FEATURE_NAMES,
    }
    print(json.dumps(metrics, indent=2))
    return model, metrics, idx_te, X_te, proba


def export_onnx(model, X_sample: np.ndarray, proba_ref: np.ndarray) -> Path:
    import onnx
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    n_features = X_sample.shape[1]
    onx = convert_xgboost(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=15,
    )
    # Strip ZipMap (if present) so the probabilities output is a plain float
    # tensor [N, 2] instead of a sequence of maps -- much nicer from Java.
    graph = onx.graph
    zipmaps = [n for n in graph.node if n.op_type == "ZipMap"]
    for zm in zipmaps:
        raw = zm.input[0]
        graph.node.remove(zm)
        for out in graph.output:
            if out.name == zm.output[0]:
                out.name = raw
                out.type.Clear()
                t = out.type.tensor_type
                t.elem_type = onnx.TensorProto.FLOAT
                t.shape.dim.add().dim_param = "N"
                t.shape.dim.add().dim_value = 2
    onnx.checker.check_model(onx)

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "fraud_xgb.onnx"
    path.write_bytes(onx.SerializeToString())

    # Parity check: ONNX Runtime must agree with XGBoost predict_proba.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": X_sample[:2048].astype(np.float32)})
    onnx_proba = np.asarray(out[1])[:, 1]
    max_diff = float(np.max(np.abs(onnx_proba - proba_ref[:2048])))
    print(f"ONNX parity check: max |diff| = {max_diff:.6f}")
    assert max_diff < 1e-3, "ONNX output diverges from XGBoost"
    print(
        f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB), "
        f"outputs: {[o.name for o in sess.get_outputs()]}"
    )
    return path


def export_stream(df: pd.DataFrame, idx_te: np.ndarray):
    """Held-out transactions, time-ordered, for live replay by the simulator."""
    DATA_DIR.mkdir(exist_ok=True)
    cols = ["card_id", "Amount", *V_COLS, "Class"]
    stream = df.iloc[np.sort(idx_te)][cols]
    path = DATA_DIR / "stream.csv.gz"
    with gzip.open(path, "wt", newline="") as f:
        stream.to_csv(f, index=False, float_format="%.6f")
    print(f"Wrote {path} ({path.stat().st_size / 1024 / 1024:.1f} MB, {len(stream):,} rows)")


def main():
    df = download()
    df = assign_cards(df)
    print("Computing per-card rolling features (offline/online parity)...")
    df = rolling_features(df)
    model, metrics, idx_te, X_te, proba = train(df)
    export_onnx(model, X_te, proba)
    export_stream(df, idx_te)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
