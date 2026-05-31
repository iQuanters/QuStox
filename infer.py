#!/usr/bin/env python3
"""
infer.py — run HQRC inference on a chosen date/time using the SAVED readout.
================================================================================
Pick one or more rows from the dataset BY TIMESTAMP, evolve each row's price
window through the fixed 16-qubit Ising quantum reservoir (local simulator), then
classify the resulting 43 features with the ALREADY-TRAINED readout weights
(readout_weights.joblib / .json produced by hqrc_nn.py). No retraining.

Flow per selected row:
    Close_prev_1..n window  ──▶  16-qubit Ising reservoir  ──▶  40 features
                                    (fixed, seeded, NOT trained)
    40 features + Open/High/Low  ──▶  43 features  ──▶  saved MLP  ──▶  buy/sell + confidence

The reservoir config (window / encode-scale / seed / shots) is read back from
readout_weights.json so the features match exactly what the readout was trained on.
This is a self-contained file (it does NOT import hqrc_nn.py).

Usage
-----
    # single bar nearest a timestamp (wall-clock, US/Eastern as shown in the CSV)
    python infer.py "2026-03-05 06:30"

    # every bar on a whole day (date only, no time)
    python infer.py 2026-03-05

    # an inclusive range of bars
    python infer.py "2026-03-05 09:30" "2026-03-05 10:30"

    # different dataset / readout
    python infer.py 2026-05-29 --dataset SPY_10m_v2.csv --readout readout_weights

Timestamps are matched against the CSV's start_ts in US/Eastern wall-clock
(the times printed in the file), so just type the time you see in the data.
================================================================================
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile

try:                                   # qiskit-aer is OPTIONAL
    from qiskit_aer import AerSimulator
    HAVE_AER = True
except ImportError:
    HAVE_AER = False                   # → base-qiskit statevector fallback

import joblib

HERE = Path(__file__).parent
MARKET_TZ = "America/New_York"         # CSV start_ts is displayed in US/Eastern


# ============================================================ quantum reservoir
class QuantumReservoir:
    """16 qubits = 4 groups of 4; intra-group all-to-all + inter-group ring (28
    edges); fixed random Ising; RX 2^k input multiplex; readout = ⟨Z_i⟩(16) +
    intra-group ⟨Z_iZ_j⟩(24) = 40.  Identical physics to hqrc_nn.py — runs on a
    local Aer simulator (or a base-qiskit statevector fallback)."""

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

        # seed the shot sampler so inference is reproducible run-to-run
        self.sim = AerSimulator(seed_simulator=seed) if HAVE_AER else "statevector"
        self._statevector = not HAVE_AER
        self._sample_rng = np.random.default_rng(seed + 12345)
        self.backend_name = "AerSimulator" if HAVE_AER else "statevector (base qiskit)"

    @property
    def output_dim(self) -> int:
        return self.n_qubits + len(self.intra_pairs)

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
        bits = np.array([[1 - 2 * int(c) for c in k.replace(" ", "")[::-1]] for k in keys],
                        dtype=np.float64)
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
        """(N, window) value windows → (N, 40) mean-state feature matrix."""
        windows = np.asarray(windows, dtype=np.float64)
        circuits = [self.build_circuit(w) for w in windows]
        tqc = circuits if self._statevector else transpile(circuits, self.sim, optimization_level=1)
        counts = self._run_counts(tqc)
        return np.stack([self._counts_to_features(c) for c in counts])


def class_confidence(p) -> np.ndarray:
    """0-100 confidence = |2·P(buy) − 1|·100  (0 = coin-flip, 100 = certain)."""
    return np.clip(np.abs(2.0 * np.asarray(p) - 1.0) * 100.0, 0.0, 100.0)


# ============================================================ data / readout
def load_readout(path: str):
    """Return (sklearn_pipeline, meta_dict). Raises if the .joblib is missing."""
    jp, bp = Path(f"{path}.json"), Path(f"{path}.joblib")
    if not bp.exists():
        raise SystemExit(f"[readout] no saved readout at {bp} — train one first by running hqrc_nn.py")
    pipe = joblib.load(bp)
    meta = json.loads(jp.read_text()) if jp.exists() else {}
    return pipe, meta


def load_dataset(path: str):
    """Load the v2 CSV; return (df, prev_cols, ohl_cols, ts_wall) where ts_wall is
    a tz-naive US/Eastern wall-clock Series aligned to df (or None if no timestamp)."""
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        p = HERE / path                          # allow bare filename in repo root
    if not p.exists():
        raise SystemExit(f"[data] dataset not found: {path}")
    df = pd.read_csv(p)

    prev_cols = sorted([c for c in df.columns if c.lower().startswith("close_prev_")],
                       key=lambda c: int(c.split("_")[-1]))
    if not prev_cols:
        raise SystemExit(f"{p}: no Close_prev_* columns found")
    ohl_cols = [c for c in ["Open", "High", "Low"] if c in df.columns]

    ts_wall = None
    ts_col = next((c for c in ["start_ts", "Datetime", "Date", "Timestamp"] if c in df.columns), None)
    if ts_col is not None:
        ts_wall = (pd.to_datetime(df[ts_col], utc=True)
                   .dt.tz_convert(MARKET_TZ).dt.tz_localize(None))
    return df, prev_cols, ohl_cols, ts_wall


# ============================================================ row selection
def _has_time(s: str) -> bool:
    """True if the user typed a clock time (so we match an exact bar, not a day)."""
    return (":" in s) or ("T" in s) or (len(s.strip()) > 10)


def select_rows(df, ts_wall, args) -> np.ndarray:
    """Resolve the CLI date/time argument(s) into integer row indices into df."""
    n = len(df)

    # explicit row-index escape hatch (works even without timestamps)
    if args.index is not None:
        idx = [i if i >= 0 else n + i for i in args.index]
        bad = [i for i in idx if not 0 <= i < n]
        if bad:
            raise SystemExit(f"[select] row index out of range: {bad} (have {n} rows)")
        return np.array(sorted(set(idx)))

    if ts_wall is None:
        raise SystemExit("[select] dataset has no timestamp column — use --index instead")

    when = args.when
    if not when:
        raise SystemExit("[select] give a date/time (e.g. \"2026-03-05 06:30\") or --index")

    # range: two args, OR --start/--end
    start_s = args.start or (when[0] if len(when) == 2 else None)
    end_s   = args.end   or (when[1] if len(when) == 2 else None)
    if start_s is not None or end_s is not None:
        lo = pd.to_datetime(start_s) if start_s else ts_wall.min()
        hi = pd.to_datetime(end_s) if end_s else ts_wall.max()
        if end_s and not _has_time(end_s):           # date-only end → include whole day
            hi = hi + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = (ts_wall >= lo) & (ts_wall <= hi)
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            raise SystemExit(f"[select] no bars in range {lo} … {hi}")
        return idx

    # single argument
    if len(when) != 1:
        raise SystemExit("[select] pass one timestamp, two for a range, or --start/--end")
    s = when[0]
    if args.all_day or not _has_time(s):             # whole-day match
        day = pd.to_datetime(s).date()
        idx = np.where((ts_wall.dt.date == day).to_numpy())[0]
        if len(idx) == 0:
            raise SystemExit(f"[select] no bars found on {day}")
        return idx

    # nearest single bar to an exact timestamp
    target = pd.to_datetime(s)
    diffs = (ts_wall - target).abs()
    i = int(diffs.values.argmin())
    gap = diffs.iloc[i]
    if gap > pd.Timedelta(minutes=0):
        print(f"[select] no exact bar at {target}; using nearest at "
              f"{ts_wall.iloc[i]} (Δ {gap})")
    return np.array([i])


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="HQRC inference on chosen date/time using saved readout weights.")
    ap.add_argument("when", nargs="*", help="one timestamp/date, or two for an inclusive range")
    ap.add_argument("--start", help="range start (overrides positional)")
    ap.add_argument("--end", help="range end (overrides positional)")
    ap.add_argument("--all-day", action="store_true", help="match the whole day even if a time is given")
    ap.add_argument("--index", type=int, nargs="+", help="select by raw row index instead of timestamp (negatives ok)")
    ap.add_argument("--dataset", default="AAPL_10m_v2.csv", help="v2 CSV (default AAPL_10m_v2.csv)")
    ap.add_argument("--readout", default="readout_weights", help="readout path stem (.joblib/.json)")
    ap.add_argument("--shots", type=int, help="override measurement shots (default from readout meta)")
    args = ap.parse_args()

    # 1. saved readout + the reservoir config it was trained with -----------------
    pipe, meta = load_readout(args.readout)
    window     = int(meta.get("reservoir_window_n", 12))
    encode_eff = float(meta.get("eff_encode_scale", 1.0))
    seed       = int(meta.get("seed", 42))
    shots      = args.shots or int(meta.get("shots", 1000))
    expect_in  = int(meta.get("n_input_features", 43))
    print(f"\n{'='*70}\n  HQRC inference — saved readout: {args.readout}.joblib\n{'='*70}")
    print(f"[readout] window={window}  encode_scale={encode_eff:.4f}  seed={seed}  "
          f"shots={shots}  expects {expect_in} features")
    if meta.get("dataset_file") and meta["dataset_file"] != args.dataset:
        print(f"[readout] note: trained on {meta['dataset_file']}, inferring on {args.dataset}")

    # 2. data + row selection ----------------------------------------------------
    df, prev_cols, ohl_cols, ts_wall = load_dataset(args.dataset)
    if len(prev_cols) != window:
        print(f"[data] WARNING: dataset has {len(prev_cols)} lags but readout expects "
              f"window={window} — features will not line up")
    rows = select_rows(df, ts_wall, args)
    print(f"[select] {len(rows)} row(s) selected from {args.dataset}")

    X   = df[prev_cols].to_numpy(dtype=np.float64)[rows]
    OHL = df[ohl_cols].to_numpy(dtype=np.float64)[rows] if ohl_cols else None

    # 3. evolve the reservoir on the selected windows ----------------------------
    res = QuantumReservoir(window=window, shots=shots, encode_scale=encode_eff, seed=seed)
    print(f"[qrc] evolving {len(rows)} window(s) on {res.backend_name} "
          f"({res.n_qubits} qubits) → {res.output_dim} features each …")
    Q = res.transform(X)
    feats = Q if OHL is None else np.hstack([Q, OHL])
    if feats.shape[1] != expect_in:
        print(f"[qrc] WARNING: built {feats.shape[1]} features but readout expects {expect_in}")

    # 4. classify with the saved readout -----------------------------------------
    mlp = pipe.named_steps["mlpclassifier"]
    buy_col = list(mlp.classes_).index(1)
    proba = pipe.predict_proba(feats)[:, buy_col]
    pred = (proba >= 0.5).astype(int)
    conf = class_confidence(proba)

    has_truth = "target" in df.columns
    if has_truth:
        y = (df["target"].astype(str).str.lower() == "buy").astype(int).to_numpy()[rows]

    # 5. report ------------------------------------------------------------------
    print("\n" + "=" * 78)
    hdr = f"  {'timestamp':<21}{'signal':>8}{'P(buy)':>9}{'conf%':>8}"
    if has_truth:
        hdr += f"{'actual':>9}{'hit':>5}"
    print(hdr)
    print("-" * 78)
    for k, r in enumerate(rows):
        when = str(ts_wall.iloc[r]) if ts_wall is not None else f"row {r}"
        sig = "BUY" if pred[k] == 1 else "SELL"
        line = f"  {when:<21}{sig:>8}{proba[k]:>9.3f}{conf[k]:>8.1f}"
        if has_truth:
            act = "buy" if y[k] == 1 else "sell"
            hit = "✓" if pred[k] == y[k] else "✗"
            line += f"{act:>9}{hit:>5}"
        print(line)
    print("=" * 78)

    if has_truth and len(rows) > 1:
        acc = float((pred == y).mean())
        print(f"  accuracy over {len(rows)} selected rows: {acc:.1%}")
    print()


if __name__ == "__main__":
    main()
