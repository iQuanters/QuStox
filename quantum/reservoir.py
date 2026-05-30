"""
Quantum Reservoir Computer (PennyLane backend).

Architecture
------------
1. Angle encoding  : RY(π·fᵢ) on qubit i          — initial state preparation
2. Reservoir layers: data re-uploading + CNOT ring  — repeated entangling + re-encoding
3. Readout         : ⟨Zᵢ⟩ + ⟨ZᵢZⱼ⟩ expectation values — captures both local and
                                                       correlated quantum features

Data Re-uploading
  Features are re-injected at EVERY layer (not just the first).
  This gives the circuit polynomial expressibility in the features,
  allowing it to capture nonlinear cross-terms that a single encoding layer
  would miss. Output_dim = n_qubits + C(n_qubits, 2).

Only the downstream linear readout is trained; all reservoir weights are fixed.
"""

import numpy as np
import pennylane as qml
from itertools import combinations


class QuantumReservoir:
    """PennyLane-backed fixed quantum reservoir with data re-uploading."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 3, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)
        # Fixed random reservoir weights — never optimized
        # Shape: (n_layers, n_qubits, 2)  — two rotation angles per qubit per layer
        self._params = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 2))

        self._pairs = list(combinations(range(n_qubits), 2))
        self._build_qnode()

    def _build_qnode(self):
        dev = qml.device("default.qubit", wires=self.n_qubits)
        n_q = self.n_qubits
        n_l = self.n_layers
        params = self._params
        pairs = self._pairs

        @qml.qnode(dev, interface="numpy")
        def circuit(features):
            # --- Initial Hadamard layer (create superposition) ---
            for i in range(n_q):
                qml.Hadamard(wires=i)

            # --- Reservoir layers with data re-uploading ---
            for layer in range(n_l):
                # Re-upload features into rotation angles
                for i in range(n_q):
                    qml.RY(np.pi * features[i], wires=i)

                # Fixed random rotations (reservoir weights)
                for i in range(n_q):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)

                # Entangling CNOT ring
                for i in range(n_q - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_q - 1, 0])

            # --- Readout: single-qubit Z + two-qubit ZZ correlators ---
            z_meas  = [qml.expval(qml.PauliZ(i)) for i in range(n_q)]
            zz_meas = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in pairs]
            return z_meas + zz_meas

        self._qnode = circuit

    @property
    def output_dim(self) -> int:
        # n_qubits single + C(n_qubits, 2) pairs
        return self.n_qubits + len(self._pairs)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n_samples, n_features), values in [-1, 1].

        Returns
        -------
        Q : (n_samples, output_dim) array of expectation values in [-1, 1].
        """
        results = []
        for x in X:
            features = np.asarray(x[: self.n_qubits], dtype=float)
            raw = self._qnode(features)
            row = [float(v) for v in raw]
            results.append(row)
        return np.array(results, dtype=np.float64)
