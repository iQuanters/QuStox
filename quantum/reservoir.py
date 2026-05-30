"""
Quantum Reservoir Computer — Qiskit backend with genuine temporal memory.

Design
------
16 qubits arranged as 4 groups of 4.

  * Intra-group : all-to-all Ising coupling (every qubit in a group of 4
                  is coupled to every other → 6 couplings per group).
  * Inter-group : a ring of bridge couplings between consecutive groups
                  (3-4, 7-8, 11-12, 15-0).

Temporal memory (the whole point of a QRC)
------------------------------------------
For each target time t we take a window of the previous `window` returns
[r_{t-window+1}, …, r_t] and let a SINGLE quantum state evolve coherently
through the entire window:

    |0…0⟩ ──[encode r_1]──[U_Ising]──[encode r_2]──[U_Ising]── … ──[measure]

There is NO mid-window measurement and NO reset, so each input perturbs the
already-evolved state. Past inputs influence the final state through the
entanglement built up by the Ising dynamics — a genuine *fading quantum
memory*. We measure only at the end (1000 shots) and average.

Encoding
--------
Univariate: the scalar return r_j is encoded on every qubit. Within each group
qubit k (k = 0..3) uses an exponential frequency multiplier 2^k:

    RX(π · 2^k · r_j)   on qubit (4·group + k)

The four groups share the same input; the fixed random Ising fields/couplings
and the ring topology break the symmetry so the groups evolve differently.

Reservoir unitary
-----------------
A fixed (seeded, never trained) random transverse-field Ising Hamiltonian,
Trotterised:  per step  Π RX(h_x)RY(h_y)RZ(h_z)  then  Π RZZ(J_z) RXX(J_x).

Readout features (per window)
-----------------------------
    ⟨Z_i⟩                  for all 16 qubits          → 16
    ⟨Z_i Z_j⟩ (intra-group) for the 6 pairs in each of → 24
                            the 4 groups
                                                  total = 40
all estimated from the same 1000-shot measurement record.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .noise import build_noise_model

_CACHE_DIR = Path(__file__).parent.parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)


class QuantumReservoir:
    """16-qubit Ising quantum reservoir with sliding-window temporal memory."""

    def __init__(
        self,
        n_groups: int = 4,
        group_size: int = 4,
        window: int = 30,
        shots: int = 1000,
        trotter_steps: int = 1,
        evo_time: float = 1.0,
        encode_scale: float = 1.0,
        return_norm: float = 0.02,
        noise: bool = True,
        seed: int = 42,
    ):
        self.n_groups = n_groups
        self.group_size = group_size
        self.n_qubits = n_groups * group_size
        self.window = window
        self.shots = shots
        self.trotter_steps = trotter_steps
        self.evo_time = evo_time
        self.encode_scale = encode_scale
        self.return_norm = return_norm
        self.noise = noise
        self.seed = seed

        # --- coupling topology ----------------------------------------------
        self.groups = [
            list(range(g * group_size, (g + 1) * group_size))
            for g in range(n_groups)
        ]
        self.intra_pairs: list[tuple[int, int]] = []
        for grp in self.groups:
            self.intra_pairs.extend(combinations(grp, 2))   # all-to-all in group
        self.inter_pairs: list[tuple[int, int]] = [
            (self.groups[g][-1], self.groups[(g + 1) % n_groups][0])
            for g in range(n_groups)
        ]                                                    # ring of groups
        self.edges = self.intra_pairs + self.inter_pairs

        # --- fixed random Ising parameters (seeded → never trained) ----------
        rng = np.random.default_rng(seed)
        self.hx = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.hy = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.hz = rng.uniform(-0.5, 0.5, self.n_qubits)
        self.Jx = rng.uniform(-1.0, 1.0, len(self.edges))
        self.Jz = rng.uniform(-1.0, 1.0, len(self.edges))

        # --- simulator -------------------------------------------------------
        # Noise is attached to native gates (rx/ry/rxx/rzz), so the circuit is
        # always simulated natively — no cx decomposition needed either way.
        noise_model = build_noise_model() if noise else None
        self.sim = AerSimulator(noise_model=noise_model)

        # readout layout
        self._z_index = list(range(self.n_qubits))
        self._zz_index = self.intra_pairs

    # ------------------------------------------------------------------ build
    @property
    def output_dim(self) -> int:
        return len(self._z_index) + len(self._zz_index)

    def _append_ising(self, qc: QuantumCircuit) -> None:
        """One fixed random-Ising evolution block (Trotterised)."""
        t = self.evo_time
        for _ in range(self.trotter_steps):
            for i in range(self.n_qubits):
                qc.rx(t * self.hx[i], i)
                qc.ry(t * self.hy[i], i)
                qc.rz(t * self.hz[i], i)
            for (i, j), jx, jz in zip(self.edges, self.Jx, self.Jz):
                qc.rzz(t * jz, i, j)
                qc.rxx(t * jx, i, j)

    def _encode(self, qc: QuantumCircuit, r: float) -> None:
        """Encode one scalar return on all qubits (2^k freq. multiplier)."""
        a = self.encode_scale * np.pi * float(r)
        for grp in self.groups:
            for k, q in enumerate(grp):
                qc.rx((2 ** k) * a, q)

    def build_circuit(self, window_returns: np.ndarray) -> QuantumCircuit:
        """Full sliding-window circuit: encode → evolve, repeated, then measure."""
        qc = QuantumCircuit(self.n_qubits)
        for r in window_returns:
            self._encode(qc, r)
            self._append_ising(qc)
        qc.measure_all()
        return qc

    # --------------------------------------------------------------- features
    def _counts_to_features(self, counts: dict[str, int]) -> np.ndarray:
        """⟨Z_i⟩ and intra-group ⟨Z_iZ_j⟩ from a shot histogram."""
        keys = list(counts.keys())
        weights = np.array([counts[k] for k in keys], dtype=np.float64)
        total = weights.sum()

        # bitstring → ±1 matrix, indexed [outcome, qubit]
        # Qiskit strings are little-endian: char (n-1-i) ↔ qubit i, so reverse.
        bits = np.array(
            [[1 - 2 * int(c) for c in k.replace(" ", "")[::-1]] for k in keys],
            dtype=np.float64,
        )  # shape (n_outcomes, n_qubits), values in {+1, -1}

        probs = weights / total
        z = probs @ bits                                  # ⟨Z_i⟩  → (n_qubits,)
        zz = np.array(
            [probs @ (bits[:, i] * bits[:, j]) for i, j in self._zz_index]
        )
        return np.concatenate([z, zz])

    def _build_windows(self, returns: np.ndarray) -> np.ndarray:
        """Stack sliding windows; left-pad the early ones with zeros."""
        n = len(returns)
        W = np.zeros((n, self.window), dtype=np.float64)
        for t in range(n):
            lo = max(0, t - self.window + 1)
            seg = returns[lo : t + 1]
            W[t, self.window - len(seg):] = seg
        return W

    def _cache_key(self, returns: np.ndarray) -> Path:
        cfg = (
            f"{self.n_qubits}_{self.window}_{self.shots}_{self.trotter_steps}_"
            f"{self.evo_time}_{self.encode_scale}_{self.return_norm}_"
            f"{int(self.noise)}_{self.seed}"
        )
        h = hashlib.sha1(
            cfg.encode() + returns.tobytes()
        ).hexdigest()[:16]
        return _CACHE_DIR / f"qrc_features_{h}.npy"

    def transform(
        self,
        returns: np.ndarray,
        use_cache: bool = True,
        batch_size: int = 200,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Map a return series → (len(returns), output_dim) quantum feature matrix.

        Each row t is the reservoir read-out after evolving through the window
        ending at t. Results are cached to disk keyed by config + data.
        """
        returns = np.asarray(returns, dtype=np.float64)
        cache_path = self._cache_key(returns)
        if use_cache and cache_path.exists():
            if verbose:
                print(f"[qrc] loaded cached features ← {cache_path.name}")
            return np.load(cache_path)

        windows = self._build_windows(returns)
        n = len(windows)
        feats = np.zeros((n, self.output_dim), dtype=np.float64)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            circuits = [self.build_circuit(windows[t]) for t in range(start, end)]
            tqc = transpile(circuits, self.sim, optimization_level=1)
            result = self.sim.run(tqc, shots=self.shots).result()
            for i in range(end - start):
                feats[start + i] = self._counts_to_features(result.get_counts(i))
            if verbose:
                print(f"[qrc] windows {end}/{n}", flush=True)

        if use_cache:
            np.save(cache_path, feats)
            if verbose:
                print(f"[qrc] cached features → {cache_path.name}")
        return feats
