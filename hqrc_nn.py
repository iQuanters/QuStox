#!/usr/bin/env python3
"""
HQRC-NN — single-file QRC + NN classifier (buy / sell).
================================================================================
16-qubit Ising Quantum Reservoir (Qiskit) + 2-layer NN classifier [43, 6, 1]
for AAPL buy/sell classification.

Per row:
  * the close_prev_1 … close_prev_n values EVOLVE the quantum reservoir
    (one coherent state through all n steps, no reset / no mid-step measurement);
  * after evolving, the reservoir is measured for SHOTS shots and the mean
    states are read out as 40 features ( ⟨Z_i⟩ × 16  +  intra-group ⟨Z_iZ_j⟩ × 24 );
  * those 40 features are concatenated with the row's Open, High, Low (→ 43) and
    fed to a 2-layer NN classifier that predicts the target: buy or sell.

Baseline: a CLASSICAL echo-state reservoir built to mirror the quantum one
(16 nodes, same group/ring topology, fixed random weights, 2^k input multiplex,
same 40-feature readout) feeds the identical NN — isolating quantum vs classical
reservoir dynamics. A majority-class baseline is also reported.

Everything needed to TRAIN and TEST lives in this one file.

Dependencies: numpy, pandas, scikit-learn (+ bundled joblib), qiskit.
qiskit-aer is OPTIONAL — without it the reservoir runs on a base-qiskit
statevector simulator (or set BACKEND to a real QPU). Only NOISE=True needs Aer.

How to run (one self-contained file, no prep step)
--------------------------------------------------
Just execute this file. It loads the dataset directly from a SINGLE CSV — the
v2 file produced by transform_aapl.py / build_dataset.py — with columns:

  * Open, High, Low                 → concatenated to the NN readout input
  * Close_prev_1 … Close_prev_n     → (N, n) window that EVOLVES the reservoir
                                       [Close_prev_1 = oldest … n = most recent]
  * target  ('buy' / 'sell')        → label  (1 = buy, 0 = sell)

No prepare_data.py / .npy step is needed. Point DATASET_FILE at the CSV (or pass
it as the one optional CLI argument). After training, the readout-layer weights
are saved (READOUT_PATH.joblib / .npz / .json) so the trained classifier can be
reloaded and applied to fresh reservoir features on the QPU without retraining.

All knobs live in the CONFIG block below — edit those values, then run.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile

try:                                  # qiskit-aer is OPTIONAL (absent on some QPU platforms)
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
        ReadoutError,
    )
    HAVE_AER = True
except ImportError:
    HAVE_AER = False                  # → fall back to a base-qiskit statevector simulator

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ----------------------------------------------------------------------- paths
HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
try:
    CACHE_DIR.mkdir(exist_ok=True)
except Exception:
    pass  # read-only filesystem (some cloud platforms) — caching just gets skipped

# native gates the reservoir is simulated with (noise attaches to these)
BASIS_GATES = ["rx", "ry", "rz", "rxx", "rzz"]

# ============================================================ CONFIG (edit here)
# Change these values and run the file directly. The dataset path may also be
# given as the single optional CLI argument (overrides DATASET_FILE).

# ── EXECUTION BACKEND ─────────────────────────────────────────────────────────
#   Default: local Aer simulator if installed, else a base-qiskit "statevector"
#   fallback that needs NO qiskit-aer (works on QPU platforms that lack Aer).
#   To run on a real IBM QPU instead, replace BACKEND below, e.g.:
#       from qiskit_ibm_runtime import QiskitRuntimeService
#       BACKEND = QiskitRuntimeService().backend("ibm_marrakesh")
#   (execution then routes through the SamplerV2 primitive automatically).
BACKEND = AerSimulator() if HAVE_AER else "statevector"
# ──────────────────────────────────────────────────────────────────────────────

# ONE self-contained CSV (Open, High, Low, Close_prev_1..n, target):
DATASET_FILE  = "AAPL_10m_v2.csv"

SAVE_READOUT  = True             # persist the trained readout layer after training
READOUT_PATH  = "readout_weights"  # writes READOUT_PATH .joblib / .npz / .json

SHOTS = 1000                     # measurement shots per window (mean states)
NOISE = False                    # realistic noise (much slower — leave False unless testing)
TEST_FRAC = 0.25                 # fraction of the (time-ordered) data held out for testing
ENCODE_SCALE = 1.0               # RX encoding scale applied to the close_prev values
RUN_CLASSICAL_BASELINE = True    # also train an NN on a CLASSICAL reservoir (matched topology)
CLASSICAL_SPECTRAL_RADIUS = 0.9  # echo-state property for the classical reservoir (fading memory)
CONF_THRESHOLD = 60.0            # only act when confidence ≥ this (0-100); low-conf rows are dropped
BATCH_SIZE = 200                 # circuits simulated per Aer batch
USE_CACHE = True                 # cache reservoir features to cache/ (instant re-runs)
SEED = 42


# ============================================================ realistic noise
def build_noise_model(
    p1: float = 1e-3, p2: float = 1e-2, p_readout: float = 1e-2,
    t1_us: float = 120.0, t2_us: float = 90.0,
    t_1q_ns: float = 50.0, t_2q_ns: float = 300.0,
) -> NoiseModel:
    """Gentle, realistic superconducting-style noise on the native gates."""
    nm = NoiseModel(basis_gates=BASIS_GATES)
    t1, t2 = t1_us * 1e3, t2_us * 1e3  # ns

    relax_1q = thermal_relaxation_error(t1, t2, t_1q_ns)
    err_1q = depolarizing_error(p1, 1).compose(relax_1q)
    nm.add_all_qubit_quantum_error(err_1q, ["rx", "ry"])

    relax_2q = thermal_relaxation_error(t1, t2, t_2q_ns).tensor(
        thermal_relaxation_error(t1, t2, t_2q_ns)
    )
    err_2q = depolarizing_error(p2, 2).compose(relax_2q)
    nm.add_all_qubit_quantum_error(err_2q, ["rxx", "rzz"])

    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
    )
    return nm


# ============================================================ quantum reservoir
class QuantumReservoir:
    """
    16 qubits = 4 groups of 4.
      intra-group : all-to-all Ising coupling (6 per group → 24)
      inter-group : ring bridges 3-4, 7-8, 11-12, 15-0 (4)        = 28 edges
    Each window value is encoded (RX, 2^k freq. multiplex per group) and a single
    state is evolved through the whole window via a fixed random Ising Hamiltonian
    (no reset, no mid-window measurement → fading quantum memory), then measured.
    Features per window: ⟨Z_i⟩ (16) + intra-group ⟨Z_iZ_j⟩ (24) = 40.
    """

    def __init__(self, window=12, n_groups=4, group_size=4, shots=1000,
                 trotter_steps=1, evo_time=1.0, encode_scale=1.0,
                 noise=False, seed=42, backend=None):
        self.n_groups, self.group_size = n_groups, group_size
        self.n_qubits = n_groups * group_size
        self.window, self.shots = window, shots
        self.trotter_steps, self.evo_time = trotter_steps, evo_time
        self.encode_scale, self.noise, self.seed = encode_scale, noise, seed

        self.groups = [list(range(g * group_size, (g + 1) * group_size))
                       for g in range(n_groups)]
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

        # use the BACKEND chosen at the top of the file (simulator or real QPU)
        self.sim = backend if backend is not None else BACKEND
        self._statevector = isinstance(self.sim, str) and self.sim == "statevector"
        self._sample_rng = np.random.default_rng(seed + 12345)  # for statevector sampling

        if self._statevector:
            self.backend_name = "statevector (base qiskit)"
        elif HAVE_AER and isinstance(self.sim, AerSimulator):
            self.backend_name = "AerSimulator"
        else:
            self.backend_name = type(self.sim).__name__

        if self.noise:
            if HAVE_AER and isinstance(self.sim, AerSimulator):
                self.sim.set_options(noise_model=build_noise_model())
            else:
                print("[qrc] note: NOISE=True ignored — needs an Aer simulator backend")
        # NOTE: on real IBM hardware, backend.run is deprecated — execution there
        # routes through the SamplerV2 primitive (transpile stays the same).

    @property
    def output_dim(self) -> int:
        return self.n_qubits + len(self.intra_pairs)

    def _append_ising(self, qc: QuantumCircuit) -> None:
        t = self.evo_time
        for _ in range(self.trotter_steps):
            for i in range(self.n_qubits):
                qc.rx(t * self.hx[i], i)
                qc.ry(t * self.hy[i], i)
                qc.rz(t * self.hz[i], i)
            for (i, j), jx, jz in zip(self.edges, self.Jx, self.Jz):
                qc.rzz(t * jz, i, j)
                qc.rxx(t * jx, i, j)

    def _encode(self, qc: QuantumCircuit, v: float) -> None:
        a = self.encode_scale * np.pi * float(v)
        for grp in self.groups:
            for k, q in enumerate(grp):
                qc.rx((2 ** k) * a, q)

    def build_circuit(self, window_values: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for v in window_values:
            self._encode(qc, v)
            self._append_ising(qc)
        qc.measure_all()
        return qc

    def _counts_to_features(self, counts: dict[str, int]) -> np.ndarray:
        keys = list(counts.keys())
        weights = np.array([counts[k] for k in keys], dtype=np.float64)
        probs = weights / weights.sum()
        # Qiskit strings are little-endian: reverse so index i ↔ qubit i.
        bits = np.array(
            [[1 - 2 * int(c) for c in k.replace(" ", "")[::-1]] for k in keys],
            dtype=np.float64,
        )
        z = probs @ bits
        zz = np.array([probs @ (bits[:, i] * bits[:, j]) for i, j in self.intra_pairs])
        return np.concatenate([z, zz])

    def _run_counts(self, tqc: list) -> list[dict]:
        """
        Execute the transpiled circuits and return one counts dict per circuit.
        Branches on the active BACKEND so the same code path serves both:
          * AerSimulator        → backend.run(...)            (fast local sim)
          * real IBM QPU/runtime → SamplerV2 primitive        (backend.run is
                                                                deprecated there)
        """
        # ① base-qiskit statevector fallback (no qiskit-aer / no QPU required)
        if self._statevector:
            from qiskit.quantum_info import Statevector
            out = []
            for c in tqc:
                cc = c.remove_final_measurements(inplace=False)   # drop measure for SV
                probs = Statevector.from_instruction(cc).probabilities()
                idx = self._sample_rng.choice(len(probs), size=self.shots, p=probs)
                vals, cnts = np.unique(idx, return_counts=True)
                nq = cc.num_qubits
                out.append({format(int(v), f"0{nq}b"): int(k) for v, k in zip(vals, cnts)})
            return out

        # ② local Aer simulator
        if HAVE_AER and isinstance(self.sim, AerSimulator):
            result = self.sim.run(tqc, shots=self.shots).result()
            return [result.get_counts(i) for i in range(len(tqc))]

        # ③ real hardware / Qiskit Runtime → SamplerV2 (imported lazily)
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=self.sim)
        job = sampler.run([(c,) for c in tqc], shots=self.shots)
        res = job.result()
        counts = []
        for r in res:
            data = r.data
            # measure_all() writes to a classical register named "meas"
            reg = data.meas if hasattr(data, "meas") else next(iter(data.values()))
            counts.append(reg.get_counts())
        return counts

    def _cache_key(self, windows: np.ndarray) -> Path:
        cfg = (f"{self.n_qubits}_{self.window}_{self.shots}_{self.trotter_steps}_"
               f"{self.evo_time}_{self.encode_scale}_{int(self.noise)}_{self.seed}")
        h = hashlib.sha1(cfg.encode() + windows.tobytes()).hexdigest()[:16]
        return CACHE_DIR / f"qrc_features_{h}.npy"

    def transform(self, windows: np.ndarray, use_cache=True,
                  batch_size=200, verbose=True) -> np.ndarray:
        """(N, window) per-row value windows → (N, 40) mean-state feature matrix."""
        windows = np.asarray(windows, dtype=np.float64)
        cache_path = self._cache_key(windows)
        if use_cache and cache_path.exists():
            if verbose:
                print(f"[qrc] loaded cached features ← {cache_path.name}")
            return np.load(cache_path)

        n = len(windows)
        feats = np.zeros((n, self.output_dim), dtype=np.float64)
        if verbose:
            print(f"[qrc] evolving {n} windows  "
                  f"({self.n_qubits} qubits, window={self.window}, "
                  f"shots={self.shots}, backend={self.backend_name}, "
                  f"noise={'ON' if self.noise else 'OFF'})")
        t0 = time.time()

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            circuits = [self.build_circuit(windows[t]) for t in range(start, end)]
            # statevector fallback handles native gates directly — no transpile needed
            tqc = circuits if self._statevector else transpile(circuits, self.sim, optimization_level=1)
            batch_counts = self._run_counts(tqc)
            for i in range(end - start):
                feats[start + i] = self._counts_to_features(batch_counts[i])

            if verbose:
                elapsed = time.time() - t0
                rate = end / elapsed
                eta = (n - end) / rate if rate > 0 else 0.0
                print(f"\r[qrc] {_progress_bar(end / n)} {end:>5}/{n} ({end/n:5.1%}) | "
                      f"{elapsed:6.1f}s elapsed | ETA {eta:6.1f}s | "
                      f"{rate:5.1f} win/s", end="", flush=True)
        if verbose:
            print()  # newline after the progress line

        if use_cache:
            try:
                np.save(cache_path, feats)
                if verbose:
                    print(f"[qrc] cached features → {cache_path.name}")
            except Exception as e:
                if verbose:
                    print(f"[qrc] cache write skipped ({e})")
        return feats


def _progress_bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ============================================================ classical reservoir
class ClassicalReservoir:
    """
    Classical echo-state reservoir, built to mirror the quantum reservoir as
    closely as possible (the no-quantum baseline):

      * 16 nodes = 4 groups of 4                          (same grouping)
      * recurrent connectivity = quantum coupling graph   (intra-group all-to-all
        + inter-group ring; 28 edges, directed random weights on each)
      * diagonal self-drive ~ the single-qubit field terms
      * fixed random weights (never trained), scaled to a target spectral radius
        so the dynamics have the echo-state property (fading memory) — the
        classical analog of the quantum reservoir's bounded unitary evolution
      * input injection mirrors the 2^k frequency multiplex per group
      * sliding-window evolution, no reset:  x_t = tanh(W_in·u_t + W·x_{t-1})
      * readout features: 16 node states + 24 intra-group products x_i·x_j = 40
        (the classical analog of ⟨Z_i⟩ + intra-group ⟨Z_iZ_j⟩)

    Produces the same 40-dim feature vector as QuantumReservoir, so the only
    difference in the pipeline is quantum vs classical reservoir dynamics.
    """

    def __init__(self, window, n_groups=4, group_size=4,
                 spectral_radius=0.9, input_scale=1.0, seed=42):
        self.n_groups, self.group_size = n_groups, group_size
        self.n_nodes = n_groups * group_size
        self.window = window

        self.groups = [list(range(g * group_size, (g + 1) * group_size))
                       for g in range(n_groups)]
        self.intra_pairs = []
        for grp in self.groups:
            self.intra_pairs.extend(combinations(grp, 2))
        self.inter_pairs = [(self.groups[g][-1], self.groups[(g + 1) % n_groups][0])
                            for g in range(n_groups)]
        self.edges = self.intra_pairs + self.inter_pairs

        rng = np.random.default_rng(seed)
        W = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for i, j in self.edges:                       # directed random coupling per edge
            W[i, j] = rng.uniform(-1.0, 1.0)
            W[j, i] = rng.uniform(-1.0, 1.0)
        np.fill_diagonal(W, rng.uniform(-0.5, 0.5, self.n_nodes))  # self-drive ~ fields
        eig = np.max(np.abs(np.linalg.eigvals(W)))    # scale for echo-state property
        if eig > 0:
            W *= spectral_radius / eig
        self.W = W

        # input weights: 2^k frequency multiplex per group (same as RX encoding)
        win = np.zeros(self.n_nodes, dtype=np.float64)
        for grp in self.groups:
            for k, node in enumerate(grp):
                win[node] = 2 ** k
        self.W_in = input_scale * win

    @property
    def output_dim(self) -> int:
        return self.n_nodes + len(self.intra_pairs)

    def _evolve(self, window_values: np.ndarray) -> np.ndarray:
        x = np.zeros(self.n_nodes, dtype=np.float64)
        for v in window_values:
            x = np.tanh(self.W_in * v + self.W @ x)   # no reset → fading memory
        zz = np.array([x[i] * x[j] for i, j in self.intra_pairs])
        return np.concatenate([x, zz])

    def transform(self, windows: np.ndarray, verbose=True) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float64)
        if verbose:
            print(f"[cres] evolving {len(windows)} windows  "
                  f"(classical ESN: {self.n_nodes} nodes, matched topology)")
        return np.stack([self._evolve(w) for w in windows])


# ============================================================ NN classifier
class NNClassifier:
    """2-layer MLP classifier [n_features, 6, 1] (buy=1 / sell=0); standardized features."""

    def __init__(self, hidden=6, alpha=1e-3, max_iter=3000, seed=42):
        self.model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu",
                          solver="adam", alpha=alpha, max_iter=max_iter,
                          early_stopping=True, n_iter_no_change=40,
                          random_state=seed),
        )

    def fit(self, X, y, verbose=True):
        self.model.fit(X, y)
        mlp = self.model.named_steps["mlpclassifier"]
        curve = mlp.loss_curve_
        stopped_early = mlp.n_iter_ < mlp.max_iter
        if verbose:
            step = max(1, len(curve) // 40)
            sampled = curve[::step]
            lo, hi = min(sampled), max(sampled)
            span = hi - lo if hi > lo else 1.0
            bars = " ▁▂▃▄▅▆▇█"
            spark = "".join(bars[int((v - lo) / span * 8)] for v in sampled)
            val = (f"  best_val_acc={mlp.best_validation_score_:.3f}"
                   if getattr(mlp, "best_validation_score_", None) is not None else "")
            print(f"[nn]  iters={mlp.n_iter_}  loss: {curve[0]:.4f} → {curve[-1]:.4f}"
                  f"{val}  {'early-stopped' if stopped_early else 'max-iter reached'}")
            print(f"[nn]  curve: {spark}")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def proba_buy(self, X) -> np.ndarray:
        """P(class == buy(1)) per row."""
        mlp = self.model.named_steps["mlpclassifier"]
        buy_col = list(mlp.classes_).index(1)
        return self.model.predict_proba(X)[:, buy_col]

    def save(self, path: str, meta: dict | None = None) -> None:
        """
        Persist the trained readout layer (StandardScaler + MLP) so it can be
        reloaded and applied to fresh reservoir features without retraining:
          * {path}.joblib : the full sklearn pipeline (load with joblib.load)
          * {path}.npz    : raw weights (scaler mean/scale, MLP coefs/intercepts)
          * {path}.json   : metadata describing the feature pipeline it expects
        """
        from joblib import dump
        pipe = self.model
        scaler = pipe.named_steps["standardscaler"]
        mlp = pipe.named_steps["mlpclassifier"]

        dump(pipe, f"{path}.joblib")

        arrays = {
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "classes": np.asarray(mlp.classes_),
        }
        for i, c in enumerate(mlp.coefs_):
            arrays[f"coef_{i}"] = c
        for i, b in enumerate(mlp.intercepts_):
            arrays[f"intercept_{i}"] = b
        np.savez(f"{path}.npz", **arrays)

        if meta is not None:
            layers = [int(mlp.coefs_[0].shape[0])] + \
                     [int(b.shape[0]) for b in mlp.intercepts_]
            meta = {**meta, "layer_sizes": layers,
                    "activation": mlp.activation, "out_activation": mlp.out_activation_}
            with open(f"{path}.json", "w") as f:
                json.dump(meta, f, indent=2)


def class_confidence(proba_buy: np.ndarray) -> np.ndarray:
    """0-100 confidence = |2·P(buy) − 1|·100  (0 = coin-flip, 100 = certain)."""
    return np.clip(np.abs(2.0 * np.asarray(proba_buy) - 1.0) * 100.0, 0.0, 100.0)


# ============================================================ classification metrics
def classification_metrics(y_true, y_pred, name) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "model": name,
        "acc": accuracy_score(y_true, y_pred),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def confidence_report(y_true, pred, conf, thresholds=(0, 20, 40, 60, 80)):
    """Accuracy & coverage when only high-confidence rows are acted on."""
    y_true = np.asarray(y_true); pred = np.asarray(pred); conf = np.asarray(conf)
    n = len(y_true)
    print(f"    {'conf ≥':>7}{'kept':>8}{'coverage':>11}{'accuracy':>11}")
    for th in thresholds:
        mask = conf >= th
        k = int(mask.sum())
        acc = accuracy_score(y_true[mask], pred[mask]) if k else float("nan")
        acc_s = f"{acc:.1%}" if k else "—"
        print(f"    {th:>7.0f}{k:>8}{k/n:>10.1%}{acc_s:>11}")
    # robust view: the most-confident slices (guaranteed non-empty coverage)
    for frac in (0.25, 0.10):
        thr = float(np.quantile(conf, 1.0 - frac))
        mask = conf >= thr
        k = int(mask.sum())
        acc = accuracy_score(y_true[mask], pred[mask]) if k else float("nan")
        print(f"    top {int(frac*100):>2}% confident (conf ≥ {thr:4.0f}): "
              f"kept {k} ({k/n:.0%})  accuracy {acc:.1%}")


# ============================================================ load dataset
def load_dataset(path: str):
    """
    Load everything the experiment needs from ONE CSV (folds in prepare_data.py):
      X   : (N, n) Close_prev_1..n  [oldest … newest]  → evolves the reservoir
      OHL : (N, 3) Open, High, Low  (optional)         → concatenated to the NN
      y   : (N,)   target  (1 = buy, 0 = sell)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"dataset CSV not found: {path}")
    df = pd.read_csv(path)

    prev_cols = sorted(
        [c for c in df.columns if c.lower().startswith("close_prev_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not prev_cols:
        raise ValueError(f"{path}: no Close_prev_* columns found")
    X = df[prev_cols].to_numpy(dtype=np.float64)            # (N, n)

    ohl_cols = [c for c in ["Open", "High", "Low"] if c in df.columns]
    OHL = df[ohl_cols].to_numpy(dtype=np.float64) if ohl_cols else None
    if OHL is None:
        print("[data] note: no Open/High/Low columns — using quantum features only")

    if "target" not in df.columns:
        raise ValueError(f"{path}: no 'target' column found")
    y = (df["target"].astype(str).str.lower() == "buy").astype(int).to_numpy()

    print(f"[data] {path}  →  X {X.shape}  |  "
          f"OHL {None if OHL is None else OHL.shape}  |  y {y.shape}  "
          f"(n lags = {len(prev_cols)})")
    return X, OHL, y


# ============================================================ main
def main():
    print(f"\n{'='*70}")
    print(f"  HQRC + NN classifier [43, 6, 1]  |  16 qubits (4×4) Ising reservoir")
    print(f"  task=buy/sell classification  shots={SHOTS}  "
          f"test_frac={TEST_FRAC}  noise={'ON' if NOISE else 'OFF'}")
    print(f"{'='*70}\n")

    # ── dataset (ONE self-contained CSV; CLI arg overrides the config) ─────────
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else DATASET_FILE
    X, OHL, y = load_dataset(dataset_file)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError(f"aapl_x must be 2-D (N, n); got {X.shape}")
    if len(y) != len(X) or (OHL is not None and len(OHL) != len(X)):
        raise ValueError("aapl_x / aapl_ohl / aapl_y length mismatch")
    N, n = X.shape
    n_buy = int(y.sum())
    print(f"[dataset] N={N}  reservoir window n={n}  "
          f"labels: buy={n_buy} ({n_buy/N:.1%})  sell={N-n_buy} ({1-n_buy/N:.1%})")

    # ── time-ordered split ────────────────────────────────────────────────────
    split = int(N * (1 - TEST_FRAC))
    yte = y[split:]
    print(f"[split] train={split}  test={N - split}")

    # ── input normalization (calibrated to the new data transform) ─────────────
    #   close_prev offsets are raw dollar gaps (std≈1.2, p99≈4.4), NOT the old
    #   tanh-normalized returns. Fed raw, RX(π·2^k·v) wraps many times (heavy
    #   aliasing) and the classical tanh saturates. Divide by a robust scale fit
    #   on the TRAIN split (p99 of |x|) so the base encoder stays within ~[-π,π];
    #   ENCODE_SCALE stays a fine-tune multiplier on top. Applied to BOTH reservoirs.
    train_scale = float(np.percentile(np.abs(X[:split]), 99)) or 1.0
    eff_scale = ENCODE_SCALE / max(train_scale, 1e-9)
    print(f"[scale] train p99(|x|)={train_scale:.3f}  →  encode_scale={eff_scale:.4f}"
          f"  (ENCODE_SCALE={ENCODE_SCALE})")

    # ── quantum reservoir: evolve close_prev windows → 40 mean-state features ─
    res = QuantumReservoir(window=n, shots=SHOTS, encode_scale=eff_scale,
                           noise=NOISE, seed=SEED)
    print(f"[qrc] output_dim={res.output_dim}  "
          f"({res.n_qubits} ⟨Z⟩ + {len(res.intra_pairs)} ⟨ZZ⟩)")
    Q = res.transform(X, use_cache=USE_CACHE, batch_size=BATCH_SIZE)

    # ── assemble NN inputs: [quantum features | Open, High, Low] ──────────────
    feats = Q if OHL is None else np.hstack([Q, OHL])
    extra = 0 if OHL is None else OHL.shape[1]
    print(f"[nn]  input features = {Q.shape[1]} quantum"
          + (f" + {extra} OHL = {feats.shape[1]}" if extra else "") )

    results = []

    # ── QRC + NN classifier ───────────────────────────────────────────────────
    print("[train] fitting NN classifier [43,6,1] on quantum+OHL features …")
    clf = NNClassifier(hidden=6, seed=SEED).fit(feats[:split], y[:split])

    # ── persist the trained readout layer (for reuse on the QPU) ──────────────
    if SAVE_READOUT:
        clf.save(READOUT_PATH, meta={
            "dataset_file": dataset_file,
            "reservoir_window_n": int(n),
            "n_quantum_features": int(Q.shape[1]),
            "n_ohl": int(extra),
            "n_input_features": int(feats.shape[1]),
            "hidden": 6,
            "encode_scale": ENCODE_SCALE,
            "eff_encode_scale": float(eff_scale),
            "train_p99_abs_x": float(train_scale),
            "shots": SHOTS, "noise": bool(NOISE), "seed": SEED,
        })
        print(f"[save] readout weights → {READOUT_PATH}.joblib / .npz / .json")

    proba = clf.proba_buy(feats[split:])
    pred = (proba >= 0.5).astype(int)
    conf = class_confidence(proba)
    results.append(classification_metrics(yte, pred, "QRC + NN (quantum)"))
    cm = confusion_matrix(yte, pred, labels=[1, 0])   # rows/cols ordered [buy, sell]
    conf_models = [("QRC + NN (quantum)", pred, conf)]

    # ── classical-reservoir baseline: identical pipeline, classical ESN ───────
    #   close_prev → classical reservoir (40 features) → [40 | OHL] → same NN.
    if RUN_CLASSICAL_BASELINE:
        cres = ClassicalReservoir(window=n, spectral_radius=CLASSICAL_SPECTRAL_RADIUS,
                                  input_scale=eff_scale, seed=SEED)
        Qc = cres.transform(X, verbose=True)
        feats_c = Qc if OHL is None else np.hstack([Qc, OHL])
        print("[train] fitting NN classifier on classical-reservoir features …")
        clf_c = NNClassifier(hidden=6, seed=SEED).fit(feats_c[:split], y[:split])
        proba_c = clf_c.proba_buy(feats_c[split:])
        pred_c = (proba_c >= 0.5).astype(int)
        conf_c = class_confidence(proba_c)
        results.append(classification_metrics(yte, pred_c, "Classical reservoir + NN"))
        conf_models.append(("Classical reservoir + NN", pred_c, conf_c))

    # ── majority-class baseline ───────────────────────────────────────────────
    maj = 1 if y[:split].mean() >= 0.5 else 0
    pred_m = np.full_like(yte, maj)
    results.append(classification_metrics(yte, pred_m, f"Majority ({'buy' if maj else 'sell'})"))

    # ── report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Model':<30}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["acc"], reverse=True):
        print(f"{r['model']:<30}{r['acc']:>9.1%} {r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>7.3f}")
    print("=" * 72)

    tp, fn, fp, tn = cm.ravel()   # labels=[1,0] → [[TP,FN],[FP,TN]]
    print("\n  Confusion matrix — QRC + NN (rows = actual, cols = predicted)")
    print(f"  {'':>10}{'pred buy':>10}{'pred sell':>11}")
    print(f"  {'act buy':>10}{tp:>10}{fn:>11}")
    print(f"  {'act sell':>10}{fp:>10}{tn:>11}")

    # ── high-confidence decisions: drop low-confidence rows, report the rest ──
    print("\n" + "=" * 72)
    print(f"  HIGH-CONFIDENCE EVALUATION — act only when confidence ≥ {CONF_THRESHOLD:.0f}")
    print("=" * 72)
    for name, p, c in conf_models:
        mask = c >= CONF_THRESHOLD
        k = int(mask.sum())
        acc = accuracy_score(yte[mask], p[mask]) if k else float("nan")
        head = (f"acted on {k}/{len(yte)} ({k/len(yte):.1%}), "
                f"accuracy {acc:.1%}" if k else
                f"no rows reached confidence ≥ {CONF_THRESHOLD:.0f}")
        print(f"\n  {name}  →  {head}")
        confidence_report(yte, p, c)

    print(f"\n  Sample predictions (last 12 test rows; act at conf ≥ {CONF_THRESHOLD:.0f}):")
    print(f"  {'actual':>8}{'pred':>8}{'conf%':>8}{'decision':>10}{'result':>9}")
    for i in range(max(0, len(yte) - 12), len(yte)):
        a = "buy" if yte[i] == 1 else "sell"
        pl = "buy" if pred[i] == 1 else "sell"
        traded = conf[i] >= CONF_THRESHOLD
        dec = "ACT" if traded else "skip"
        ok = ("✓" if pred[i] == yte[i] else "✗") if traded else "—"
        print(f"  {a:>8}{pl:>8}{conf[i]:>7.1f}{dec:>10}{ok:>9}")
    print()

    return results


if __name__ == "__main__":
    main()
