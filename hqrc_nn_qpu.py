#!/usr/bin/env python3
"""
HQRC-NN (QPU edition) — run the quantum reservoir on a REAL IBM quantum computer.
================================================================================
Same 16-qubit Ising reservoir + NN classifier as hqrc_nn.py, but the reservoir
circuits execute on real IBM hardware via Qiskit Runtime's SamplerV2 primitive.

Differences vs hqrc_nn.py
-------------------------
  * Backend = a real IBM QPU (QiskitRuntimeService).  No qiskit-aer needed.
  * NO disk caching anywhere (hardware/submission filesystems are read-only).
  * MAX_ROWS cap — you CANNOT send thousands of circuits to a real device; this
    runs only the most recent MAX_ROWS windows (one Runtime job, chunked).
  * Default mode = INFERENCE with the readout you already trained on the
    simulator (readout_weights.* from hqrc_nn.py): the QPU only produces the
    reservoir features, the trained NN maps them to buy/sell. The reservoir
    config (seed / window / encode-scale) is read back from readout_weights.json
    so the QPU features match what the readout was trained on.
    If no saved readout is found, it trains a fresh NN on the (capped) QPU
    features instead — self-contained but statistically weak on a small subset.

Auth
----
Set your key once (either is fine):
  * env var:   export IBM_QUANTUM_TOKEN=...      (read automatically below), or
  * saved:     QiskitRuntimeService.save_account(channel=..., token=...)  once.

Dependencies: numpy, pandas, scikit-learn (+ joblib), qiskit, qiskit-ibm-runtime.
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Batch

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

HERE = Path(__file__).parent

# ============================================================ CONFIG (edit here)
# ── IBM Quantum credentials ───────────────────────────────────────────────────
IBM_TOKEN    = os.environ.get("IBM_QUANTUM_TOKEN", "")  # set the env var (do NOT hardcode/commit your key)
# With qiskit-ibm-runtime ≥0.30 the classic "ibm_quantum" channel is RETIRED.
# Use "ibm_quantum_platform" (default) or "ibm_cloud" depending on your account.
IBM_CHANNEL  = "ibm_quantum_platform"
IBM_INSTANCE = None              # IBM Cloud instance CRN if your account needs one; None = default
BACKEND_NAME = "ibm_marrakesh"   # set "" to auto-pick the least-busy real device

# ── data + trained readout ────────────────────────────────────────────────────
DATASET_FILE = "AAPL_10m_v2.csv"   # Open, High, Low, Close_prev_1..n, target
READOUT_PATH = "readout_weights"   # .joblib/.json produced by hqrc_nn.py (sim training)
USE_PRETRAINED_READOUT = True      # True → infer with the sim-trained NN (recommended)

# ── QPU run size (KEEP SMALL — real hardware!) ────────────────────────────────
MAX_ROWS  = 256        # only the most recent MAX_ROWS windows are sent to the QPU
JOB_CHUNK = 128        # circuits per Runtime job (split MAX_ROWS into chunks)
SHOTS     = 1000

# ── reservoir params (used only if NOT loading a pretrained readout) ──────────
ENCODE_SCALE   = 1.0
TEST_FRAC      = 0.25
CONF_THRESHOLD = 60.0
SEED           = 42


# ============================================================ quantum reservoir
class QuantumReservoir:
    """16 qubits = 4 groups of 4; intra-group all-to-all + inter-group ring (28
    edges); fixed random Ising; RX 2^k input multiplex; readout = ⟨Z_i⟩(16) +
    intra-group ⟨Z_iZ_j⟩(24) = 40.  Identical physics to hqrc_nn.py."""

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

    def transform_on_qpu(self, windows, backend, shots, job_chunk):
        """Build circuits, transpile to the backend ISA, run via SamplerV2 (chunked
        into Runtime jobs), and return the (N, 40) feature matrix. No caching."""
        windows = np.asarray(windows, dtype=np.float64)
        n = len(windows)
        circuits = [self.build_circuit(windows[t]) for t in range(n)]

        print(f"[qpu] transpiling {n} circuits to {backend.name} ISA …")
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa = pm.run(circuits)

        feats = np.zeros((n, self.output_dim), dtype=np.float64)
        print(f"[qpu] submitting {n} circuits in chunks of {job_chunk} "
              f"({shots} shots each) — this queues on real hardware …")
        with Batch(backend=backend) as batch:
            sampler = Sampler(mode=batch)
            for start in range(0, n, job_chunk):
                end = min(start + job_chunk, n)
                job = sampler.run([(c,) for c in isa[start:end]], shots=shots)
                print(f"[qpu]   job {job.job_id()} : circuits {start}-{end-1} submitted")
                result = job.result()
                for i, pub in enumerate(result):
                    data = pub.data
                    reg = data.meas if hasattr(data, "meas") else next(iter(data.values()))
                    feats[start + i] = self._counts_to_features(reg.get_counts())
                print(f"[qpu]   chunk {start}-{end-1} done")
        return feats


# ============================================================ NN classifier
class NNClassifier:
    """2-layer MLP [n,6,1] (buy=1/sell=0); standardized features."""
    def __init__(self, hidden=6, alpha=1e-3, max_iter=3000, seed=42):
        self.model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                          alpha=alpha, max_iter=max_iter, early_stopping=True,
                          n_iter_no_change=40, random_state=seed),
        )

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def predict(self, X):
        return self.model.predict(X)

    def proba_buy(self, X):
        mlp = self.model.named_steps["mlpclassifier"]
        return self.model.predict_proba(X)[:, list(mlp.classes_).index(1)]


def class_confidence(p):
    return np.clip(np.abs(2.0 * np.asarray(p) - 1.0) * 100.0, 0.0, 100.0)


# ============================================================ data / readout
def load_dataset(path):
    df = pd.read_csv(path)
    prev = sorted([c for c in df.columns if c.lower().startswith("close_prev_")],
                  key=lambda c: int(c.split("_")[-1]))
    if not prev:
        raise ValueError(f"{path}: no Close_prev_* columns")
    X = df[prev].to_numpy(dtype=np.float64)
    ohl = [c for c in ["Open", "High", "Low"] if c in df.columns]
    OHL = df[ohl].to_numpy(dtype=np.float64) if ohl else None
    if "target" not in df.columns:
        raise ValueError(f"{path}: no 'target' column")
    y = (df["target"].astype(str).str.lower() == "buy").astype(int).to_numpy()
    return X, OHL, y


def load_pretrained_readout(path):
    """Return (sklearn_pipeline, meta_dict) or (None, None) if absent."""
    jp, bp = Path(f"{path}.json"), Path(f"{path}.joblib")
    if not bp.exists():
        return None, None
    from joblib import load
    meta = json.loads(jp.read_text()) if jp.exists() else {}
    return load(bp), meta


# ============================================================ main
def main():
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else DATASET_FILE
    print(f"\n{'='*70}\n  HQRC-NN on REAL IBM QPU  |  16-qubit Ising reservoir\n{'='*70}\n")

    # 1. connect to IBM Quantum -------------------------------------------------
    svc_kwargs = {"channel": IBM_CHANNEL}
    if IBM_TOKEN:
        svc_kwargs["token"] = IBM_TOKEN
    if IBM_INSTANCE:
        svc_kwargs["instance"] = IBM_INSTANCE
    try:
        service = QiskitRuntimeService(**svc_kwargs)
    except Exception as e:
        raise SystemExit(f"[auth] could not init QiskitRuntimeService: {e}\n"
                         f"       set IBM_QUANTUM_TOKEN or run QiskitRuntimeService.save_account(...)")
    backend = (service.backend(BACKEND_NAME) if BACKEND_NAME
               else service.least_busy(operational=True, simulator=False))
    print(f"[qpu] backend = {backend.name}  ({backend.num_qubits} qubits)")

    # 2. data -------------------------------------------------------------------
    X, OHL, y = load_dataset(dataset_file)
    N_all = len(X)
    print(f"[data] {dataset_file}: {N_all} rows, n_lags={X.shape[1]}, "
          f"OHL={'yes' if OHL is not None else 'no'}")

    # 3. readout: load sim-trained NN (preferred) or plan to train on subset ----
    clf, meta = (load_pretrained_readout(READOUT_PATH) if USE_PRETRAINED_READOUT else (None, None))
    if clf is not None:
        window      = int(meta.get("reservoir_window_n", X.shape[1]))
        encode_eff  = float(meta.get("eff_encode_scale", ENCODE_SCALE))
        seed        = int(meta.get("seed", SEED))
        print(f"[readout] loaded sim-trained NN from {READOUT_PATH}.joblib "
              f"(window={window}, encode_scale={encode_eff:.4f}, seed={seed})")
        mode = "infer"
    else:
        window, seed = X.shape[1], SEED
        encode_eff = None   # calibrated below from the subset's train split
        mode = "train"
        print(f"[readout] no saved readout found → will TRAIN on the capped QPU subset")

    # 4. cap to the most recent MAX_ROWS windows (real hardware can't take more) -
    if N_all > MAX_ROWS:
        X, y = X[-MAX_ROWS:], y[-MAX_ROWS:]
        OHL = None if OHL is None else OHL[-MAX_ROWS:]
        print(f"[qpu] capping to the most recent {MAX_ROWS} rows (of {N_all})")
    N = len(X)

    if encode_eff is None:                      # train-mode calibration (train split only)
        split0 = int(N * (1 - TEST_FRAC))
        train_scale = float(np.percentile(np.abs(X[:split0]), 99)) or 1.0
        encode_eff = ENCODE_SCALE / max(train_scale, 1e-9)
        print(f"[scale] train p99(|x|)={train_scale:.3f} → encode_scale={encode_eff:.4f}")

    # 5. reservoir features ON THE QPU ------------------------------------------
    res = QuantumReservoir(window=window, shots=SHOTS, encode_scale=encode_eff, seed=seed)
    Q = res.transform_on_qpu(X, backend=backend, shots=SHOTS, job_chunk=JOB_CHUNK)
    feats = Q if OHL is None else np.hstack([Q, OHL])
    print(f"[nn] feature matrix = {feats.shape}")

    # 6. predict (infer) or train+test on the capped subset ---------------------
    if mode == "infer":
        proba = clf.predict_proba(feats)[:, list(clf.named_steps["mlpclassifier"].classes_).index(1)]
        pred = (proba >= 0.5).astype(int)
        conf = class_confidence(proba)
        yte = y
        title = "QPU inference (sim-trained readout)"
    else:
        split = int(N * (1 - TEST_FRAC))
        clf = NNClassifier(seed=SEED).fit(feats[:split], y[:split])
        proba = clf.proba_buy(feats[split:])
        pred = (proba >= 0.5).astype(int)
        conf = class_confidence(proba)
        yte = y[split:]
        title = f"QPU train/test on capped subset (train={split}, test={len(yte)})"

    # 7. report -----------------------------------------------------------------
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="binary",
                                                  pos_label=1, zero_division=0)
    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred, labels=[1, 0])
    print("\n" + "=" * 70)
    print(f"  {title}")
    print(f"  rows evaluated: {len(yte)}   shots/circuit: {SHOTS}")
    print("-" * 70)
    print(f"  accuracy {acc:.1%}   precision {p:.3f}   recall {r:.3f}   F1 {f1:.3f}")
    tp, fn, fp, tn = cm.ravel()
    print(f"  confusion  [act buy → pred buy {tp}, pred sell {fn}]")
    print(f"             [act sell→ pred buy {fp}, pred sell {tn}]")

    mask = conf >= CONF_THRESHOLD
    k = int(mask.sum())
    hi = accuracy_score(yte[mask], pred[mask]) if k else float("nan")
    print(f"  high-confidence (≥{CONF_THRESHOLD:.0f}): acted {k}/{len(yte)} "
          f"({k/len(yte):.1%}), accuracy {hi:.1%}" if k else
          f"  high-confidence (≥{CONF_THRESHOLD:.0f}): no rows reached threshold")
    print("=" * 70)
    print("\n  NOTE: results on a small QPU subset are not statistically significant;")
    print("        this run demonstrates the reservoir executing on real hardware.\n")


if __name__ == "__main__":
    main()
