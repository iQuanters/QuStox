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

Dependencies: numpy, scikit-learn, qiskit, qiskit-aer.

How to run (no command line / no arguments needed)
--------------------------------------------------
Just execute this file. It loads the dataset directly from three .npy files in
the run directory (produced by prepare_data.py):

  * aapl_x.npy   : (N, n) close_prev windows   [oldest … newest]  → reservoir
  * aapl_ohl.npy : (N, 3) Open, High, Low                         → NN (concatenated)
  * aapl_y.npy   : (N,)   labels   1 = buy, 0 = sell              → target

then splits it (time-ordered) into train / test. All knobs live in the CONFIG
block below — edit those values, then run.
================================================================================
"""

from __future__ import annotations

import hashlib
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

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
# No command line is used — change these values and run the file directly.
# Dataset files (referenced directly by name, relative to the run directory):
AAPL_X_FILE   = "aapl_x.npy"     # (N, n) close_prev windows  → evolve the reservoir
AAPL_OHL_FILE = "aapl_ohl.npy"   # (N, 3) Open, High, Low      → concatenated to NN input
AAPL_Y_FILE   = "aapl_y.npy"     # (N,)   labels: 1 = buy, 0 = sell

SHOTS = 1000                     # measurement shots per window (mean states)
NOISE = False                    # realistic noise (much slower — leave False unless testing)
TEST_FRAC = 0.25                 # fraction of the (time-ordered) data held out for testing
ENCODE_SCALE = 1.0               # RX encoding scale applied to the close_prev values
RUN_CLASSICAL_BASELINE = True    # also train an NN on a CLASSICAL reservoir (matched topology)
CLASSICAL_SPECTRAL_RADIUS = 0.9  # echo-state property for the classical reservoir (fading memory)
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
                 noise=False, seed=42):
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

        self.sim = AerSimulator(noise_model=build_noise_model() if noise else None)

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
                  f"shots={self.shots}, noise={'ON' if self.noise else 'OFF'})")
        t0 = time.time()

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            circuits = [self.build_circuit(windows[t]) for t in range(start, end)]
            tqc = transpile(circuits, self.sim, optimization_level=1)
            result = self.sim.run(tqc, shots=self.shots).result()
            for i in range(end - start):
                feats[start + i] = self._counts_to_features(result.get_counts(i))

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


# ============================================================ load dataset
def _load_labels(path: str) -> np.ndarray:
    y = np.load(path, allow_pickle=True).ravel()
    if y.dtype.kind in "US":            # strings like 'buy' / 'sell'
        return (np.char.lower(y.astype(str)) == "buy").astype(int)
    return y.astype(int)


def load_aapl():
    """Load X (reservoir input), OHL (extra NN features, optional), y (labels)."""
    X = np.asarray(np.load(AAPL_X_FILE), dtype=np.float64)
    y = _load_labels(AAPL_Y_FILE)
    if Path(AAPL_OHL_FILE).exists():
        OHL = np.asarray(np.load(AAPL_OHL_FILE), dtype=np.float64)
        if OHL.ndim == 1:
            OHL = OHL.reshape(-1, 1)
    else:
        OHL = None
        print(f"[data] note: {AAPL_OHL_FILE} not found — using quantum features only")
    print(f"[data] {AAPL_X_FILE} {X.shape}  |  "
          f"{AAPL_OHL_FILE} {None if OHL is None else OHL.shape}  |  "
          f"{AAPL_Y_FILE} {y.shape}")
    return X, OHL, y


# ============================================================ main
def main():
    print(f"\n{'='*70}")
    print(f"  HQRC + NN classifier [43, 6, 1]  |  16 qubits (4×4) Ising reservoir")
    print(f"  task=buy/sell classification  shots={SHOTS}  "
          f"test_frac={TEST_FRAC}  noise={'ON' if NOISE else 'OFF'}")
    print(f"{'='*70}\n")

    # ── dataset ──────────────────────────────────────────────────────────────
    X, OHL, y = load_aapl()
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
    proba = clf.proba_buy(feats[split:])
    pred = (proba >= 0.5).astype(int)
    conf = class_confidence(proba)
    results.append(classification_metrics(yte, pred, "QRC + NN (quantum)"))
    cm = confusion_matrix(yte, pred, labels=[1, 0])   # rows/cols ordered [buy, sell]

    # ── classical-reservoir baseline: identical pipeline, classical ESN ───────
    #   close_prev → classical reservoir (40 features) → [40 | OHL] → same NN.
    if RUN_CLASSICAL_BASELINE:
        cres = ClassicalReservoir(window=n, spectral_radius=CLASSICAL_SPECTRAL_RADIUS,
                                  input_scale=eff_scale, seed=SEED)
        Qc = cres.transform(X, verbose=True)
        feats_c = Qc if OHL is None else np.hstack([Qc, OHL])
        print("[train] fitting NN classifier on classical-reservoir features …")
        clf_c = NNClassifier(hidden=6, seed=SEED).fit(feats_c[:split], y[:split])
        pred_c = clf_c.predict(feats_c[split:])
        results.append(classification_metrics(yte, pred_c, "Classical reservoir + NN"))

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

    print("\n  Sample predictions (last 10 test rows):")
    print(f"  {'actual':>8}{'pred':>8}{'conf%':>8}{'result':>9}")
    for i in range(max(0, len(yte) - 10), len(yte)):
        a = "buy" if yte[i] == 1 else "sell"
        p = "buy" if pred[i] == 1 else "sell"
        ok = "✓" if pred[i] == yte[i] else "✗"
        print(f"  {a:>8}{p:>8}{conf[i]:>7.1f}{ok:>9}")
    print()

    return results


if __name__ == "__main__":
    main()
