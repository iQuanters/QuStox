#!/usr/bin/env python3
"""
infer_pipeline.py — Full HQRC inference with sentiment overlay.
================================================================
Runs Sanket's quantum inference (infer.py logic) then overlays
Parth's sentiment agent to produce a final risk-adjusted signal.

At inference time the reservoir window is FIXED to what the model
was trained on (readout_weights.json → reservoir_window_n=12).
Sentiment cannot change the window here — instead it acts as a
confidence gate and position-size multiplier:

    CALM        → accept signals at conf ≥ 40%,  size = 1.0×
    NORMAL      → accept signals at conf ≥ 50%,  size = 1.0×
    ELEVATED    → accept signals at conf ≥ 60%,  size = 0.7×
    CRISIS      → accept signals at conf ≥ 75%,  size = 0.4×
    ELECTION    → accept signals at conf ≥ 65%,  size = 0.6×
    FED_EVENT   → accept signals at conf ≥ 65%,  size = 0.5×
    GEOPOLITICAL→ accept signals at conf ≥ 70%,  size = 0.55×
    PANDEMIC    → accept signals at conf ≥ 80%,  size = 0.3×
    EARNINGS    → accept signals at conf ≥ 55%,  size = 0.8×

Usage
-----
    python infer_pipeline.py                       # most recent bar in dataset
    python infer_pipeline.py 2026-04-30            # last bar of that day (market close)
    python infer_pipeline.py "2026-04-30 09:30"    # bar nearest that exact time
    python infer_pipeline.py 2026-04-30 --no-sentiment
    python infer_pipeline.py 2026-04-30 --ticker AAPL
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from qiskit import QuantumCircuit, transpile

try:
    from qiskit_aer import AerSimulator
    HAVE_AER = True
except ImportError:
    HAVE_AER = False

HERE = Path(__file__).parent
MARKET_TZ = "America/New_York"

# ── Sentiment confidence gates + position sizes ───────────────────────────────
_REGIME_GATE: dict[str, float] = {
    "CALM":          0.40,
    "NORMAL":        0.50,
    "ELEVATED":      0.60,
    "CRISIS":        0.75,
    "ELECTION":      0.65,
    "FED_EVENT":     0.65,
    "GEOPOLITICAL":  0.70,
    "PANDEMIC":      0.80,
    "EARNINGS":      0.55,
}
_REGIME_SIZE: dict[str, float] = {
    "CALM":          1.00,
    "NORMAL":        1.00,
    "ELEVATED":      0.70,
    "CRISIS":        0.40,
    "ELECTION":      0.60,
    "FED_EVENT":     0.50,
    "GEOPOLITICAL":  0.55,
    "PANDEMIC":      0.30,
    "EARNINGS":      0.80,
}

# ── Optional sentiment agent ──────────────────────────────────────────────────
try:
    from src.agents.external_factors import get_external_factors as _get_ext
    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False


# ══════════════════════════════════════════════════ quantum reservoir (verbatim)
class QuantumReservoir:
    def __init__(self, window=12, n_groups=4, group_size=4, shots=1000,
                 trotter_steps=1, evo_time=1.0, encode_scale=1.0, seed=42):
        self.n_groups, self.group_size = n_groups, group_size
        self.n_qubits = n_groups * group_size
        self.window, self.shots = window, shots
        self.trotter_steps, self.evo_time, self.encode_scale = trotter_steps, evo_time, encode_scale

        self.groups = [list(range(g * group_size, (g + 1) * group_size)) for g in range(n_groups)]
        self.intra_pairs = []
        for grp in self.groups:
            self.intra_pairs.extend(combinations(grp, 2))
        self.inter_pairs = [(self.groups[g][-1], self.groups[(g + 1) % n_groups][0])
                            for g in range(n_groups)]
        self.edges = self.intra_pairs + self.inter_pairs

        rng = np.random.default_rng(seed)
        self.hx = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.hy = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.hz = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.Jx = rng.uniform(-1.0, 1.0, len(self.edges))
        self.Jz = rng.uniform(-1.0, 1.0, len(self.edges))

        self.sim = AerSimulator(seed_simulator=seed) if HAVE_AER else "statevector"
        self._statevector = not HAVE_AER
        self._sample_rng = np.random.default_rng(seed + 12345)
        self.backend_name = "AerSimulator" if HAVE_AER else "statevector"

    @property
    def output_dim(self): return self.n_qubits + len(self.intra_pairs)

    def _append_ising(self, qc):
        t = self.evo_time
        for _ in range(self.trotter_steps):
            for i in range(self.n_qubits):
                qc.rx(t * self.hx[i], i); qc.ry(t * self.hy[i], i); qc.rz(t * self.hz[i], i)
            for (i, j), jx, jz in zip(self.edges, self.Jx, self.Jz):
                qc.rzz(t * jz, i, j); qc.rxx(t * jx, i, j)

    def _encode(self, qc, v):
        a = self.encode_scale * np.pi * float(v)
        for grp in self.groups:
            for k, q in enumerate(grp):
                qc.rx((2 ** k) * a, q)

    def build_circuit(self, window_values):
        qc = QuantumCircuit(self.n_qubits)
        for v in window_values:
            self._encode(qc, v); self._append_ising(qc)
        qc.measure_all()
        return qc

    def _counts_to_features(self, counts):
        keys = list(counts.keys())
        w = np.array([counts[k] for k in keys], dtype=np.float64)
        probs = w / w.sum()
        bits = np.array([[1 - 2 * int(c) for c in k.replace(" ", "")[::-1]] for k in keys], dtype=np.float64)
        z = probs @ bits
        zz = np.array([probs @ (bits[:, i] * bits[:, j]) for i, j in self.intra_pairs])
        return np.concatenate([z, zz])

    def _run_counts(self, tqc):
        if self._statevector:
            from qiskit.quantum_info import Statevector
            out = []
            for c in tqc:
                cc = c.remove_final_measurements(inplace=False)
                probs = Statevector.from_instruction(cc).probabilities()
                idx = self._sample_rng.choice(len(probs), size=self.shots, p=probs)
                vals, cnts = np.unique(idx, return_counts=True)
                nq = cc.num_qubits
                out.append({format(int(v), f"0{nq}b"): int(k) for v, k in zip(vals, cnts)})
            return out
        result = self.sim.run(tqc, shots=self.shots).result()
        return [result.get_counts(i) for i in range(len(tqc))]

    def transform(self, windows) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float64)
        circuits = [self.build_circuit(w) for w in windows]
        tqc = circuits if self._statevector else transpile(circuits, self.sim, optimization_level=1)
        counts = self._run_counts(tqc)
        return np.stack([self._counts_to_features(c) for c in counts])


# ══════════════════════════════════════════════════════════ data helpers
def load_readout(path: str):
    jp, bp = Path(f"{path}.json"), Path(f"{path}.joblib")
    if not bp.exists():
        raise SystemExit(f"[readout] no weights at {bp} — run hqrc_nn.py first")
    pipe = joblib.load(bp)
    meta = json.loads(jp.read_text()) if jp.exists() else {}
    return pipe, meta


def load_dataset(path: str):
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        p = HERE / path
    if not p.exists():
        raise SystemExit(f"[data] not found: {path}")
    df = pd.read_csv(p)
    prev_cols = sorted([c for c in df.columns if c.lower().startswith("close_prev_")],
                       key=lambda c: int(c.split("_")[-1]))
    ohl_cols = [c for c in ["Open", "High", "Low"] if c in df.columns]
    ts_wall = None
    ts_col = next((c for c in ["start_ts", "Datetime", "Date"] if c in df.columns), None)
    if ts_col:
        ts_wall = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    return df, prev_cols, ohl_cols, ts_wall


def select_rows(df, ts_wall, when: list[str], index: list[int] | None) -> np.ndarray:
    """Always returns exactly ONE row — the single bar to predict on."""
    n = len(df)

    # explicit row index
    if index is not None:
        i = index[0] if index[0] >= 0 else n + index[0]
        return np.array([i])

    # no argument → most recent bar in dataset
    if ts_wall is None or not when:
        return np.array([n - 1])

    s = when[0]
    has_time = (":" in s) or len(s.strip()) > 10

    if has_time:
        # exact timestamp → nearest bar
        target = pd.to_datetime(s)
        i = int((ts_wall - target).abs().values.argmin())
    else:
        # date only → last bar of that day (market close)
        day = pd.to_datetime(s).date()
        day_idx = np.where((ts_wall.dt.date == day).to_numpy())[0]
        if len(day_idx) == 0:
            raise SystemExit(f"[select] no bars found on {day}")
        i = int(day_idx[-1])   # last bar = market close

    return np.array([i])


# ══════════════════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser(description="HQRC inference + sentiment overlay")
    ap.add_argument("when", nargs="*", help="timestamp / date / range")
    ap.add_argument("--index", type=int, nargs="+", help="select by row index")
    ap.add_argument("--dataset", default="AAPL_10m_v2.csv")
    ap.add_argument("--readout", default="readout_weights")
    ap.add_argument("--ticker", default="AAPL", help="ticker for sentiment agent")
    ap.add_argument("--shots", type=int)
    ap.add_argument("--no-sentiment", action="store_true")
    args = ap.parse_args()

    # ── 1. Load readout weights ───────────────────────────────────────────────
    pipe, meta = load_readout(args.readout)
    window     = int(meta.get("reservoir_window_n", 12))
    encode_eff = float(meta.get("eff_encode_scale", 1.0))
    seed       = int(meta.get("seed", 42))
    shots      = args.shots or int(meta.get("shots", 1000))
    expect_in  = int(meta.get("n_input_features", 43))

    print(f"\n{'='*70}")
    print(f"  QuStox · HQRC Inference Pipeline")
    print(f"  readout={args.readout}  window={window}  shots={shots}")
    print(f"{'='*70}\n")

    # ── 2. Sentiment agent ────────────────────────────────────────────────────
    regime       = "NORMAL"
    conf_gate    = _REGIME_GATE["NORMAL"]
    pos_size     = _REGIME_SIZE["NORMAL"]
    sent_score   = 0.0
    sent_reason  = "sentiment skipped"

    if _SENTIMENT_AVAILABLE and not args.no_sentiment:
        try:
            factors     = _get_ext(args.ticker, as_of=None, verbose=True)
            regime      = factors.regime
            conf_gate   = _REGIME_GATE.get(regime, 0.50)
            pos_size    = _REGIME_SIZE.get(regime, 1.00)
            sent_score  = factors.sentiment_score
            sent_reason = factors.reasoning
            print(f"\n[sentiment] regime={regime}  conf_gate≥{conf_gate:.0%}  "
                  f"pos_size={pos_size:.2f}×  sentiment={sent_score:+.2f}")
            print(f"  {sent_reason}\n")
        except Exception as exc:
            print(f"[sentiment] WARNING: failed ({exc}), using NORMAL defaults\n")
    else:
        print("[sentiment] skipped — using NORMAL defaults\n")

    # ── 3. Load data + select rows ────────────────────────────────────────────
    df, prev_cols, ohl_cols, ts_wall = load_dataset(args.dataset)
    rows = select_rows(df, ts_wall, args.when, args.index)
    print(f"[data] {len(rows)} bar(s) selected from {args.dataset}")

    X   = df[prev_cols].to_numpy(dtype=np.float64)[rows]
    OHL = df[ohl_cols].to_numpy(dtype=np.float64)[rows] if ohl_cols else None

    # ── 4. Quantum reservoir ──────────────────────────────────────────────────
    res = QuantumReservoir(window=window, shots=shots, encode_scale=encode_eff, seed=seed)
    print(f"[qrc] {res.backend_name} · {res.n_qubits} qubits · {res.output_dim} features")
    Q     = res.transform(X)
    feats = Q if OHL is None else np.hstack([Q, OHL])

    # ── 5. Classify ───────────────────────────────────────────────────────────
    mlp      = pipe.named_steps["mlpclassifier"]
    buy_col  = list(mlp.classes_).index(1)
    proba    = pipe.predict_proba(feats)[:, buy_col]
    raw_pred = (proba >= 0.5).astype(int)

    # Apply sentiment confidence gate → positions below gate become HOLD
    gated_pred = np.where(proba >= conf_gate, raw_pred,
                 np.where(proba <= (1 - conf_gate), raw_pred, -1))  # -1 = HOLD

    has_truth = "target" in df.columns
    if has_truth:
        y = (df["target"].astype(str).str.lower() == "buy").astype(int).to_numpy()[rows]

    # ── 6. Print results ──────────────────────────────────────────────────────
    print(f"\n{'='*82}")
    hdr = f"  {'timestamp':<21}{'raw':>6}{'P(buy)':>9}{'conf%':>8}{'gated':>8}{'size':>7}"
    if has_truth:
        hdr += f"{'actual':>8}{'hit':>5}"
    print(hdr)
    print(f"  {'regime: ' + regime:<21}{'gate:':>6}{conf_gate:>8.0%}{'':>8}{'pos:':>8}{pos_size:>6.2f}×")
    print("-" * 82)

    for k, r in enumerate(rows):
        ts   = str(ts_wall.iloc[r]) if ts_wall is not None else f"row {r}"
        raw  = "BUY" if raw_pred[k] == 1 else "SELL"
        conf_pct = abs(2 * proba[k] - 1) * 100
        gated_label = ("BUY" if gated_pred[k] == 1
                       else "SELL" if gated_pred[k] == 0
                       else "HOLD")
        line = f"  {ts:<21}{raw:>6}{proba[k]:>9.3f}{conf_pct:>8.1f}{gated_label:>8}{pos_size:>7.2f}"
        if has_truth:
            act = "buy" if y[k] == 1 else "sell"
            hit = "✓" if raw_pred[k] == y[k] else "✗"
            line += f"{act:>8}{hit:>5}"
        print(line)

    print("=" * 82)
    if has_truth and len(rows) > 1:
        acc = float((raw_pred == y).mean())
        print(f"  raw accuracy over {len(rows)} bars: {acc:.1%}")
    print(f"  sentiment: {regime} · pos_size={pos_size:.2f}× · score={sent_score:+.2f}\n")


if __name__ == "__main__":
    main()
