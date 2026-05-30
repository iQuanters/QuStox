"""
Gate matrices and circuit utilities for the XpyQ-backed reservoir.
All gates use only real arithmetic (RY + CNOT) so the full state vector stays real.
"""

import numpy as np

I2 = np.eye(2, dtype=np.float64)
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
X_gate = np.array([[0.0, 1.0], [1.0, 0.0]])


def RY(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def embed_single(gate_2x2: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """Lift a 2×2 gate to the full 2^n space."""
    ops = [gate_2x2 if i == target else I2 for i in range(n_qubits)]
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def embed_cnot(control: int, target: int, n_qubits: int) -> np.ndarray:
    """Build CNOT matrix in the full 2^n computational basis."""
    dim = 2 ** n_qubits
    mat = np.eye(dim, dtype=np.float64)
    for basis in range(dim):
        bits = [(basis >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
        if bits[control] == 1:
            flipped = list(bits)
            flipped[target] ^= 1
            new_basis = sum(flipped[i] << (n_qubits - 1 - i) for i in range(n_qubits))
            mat[new_basis, basis] = 1.0
            mat[basis, basis] = 0.0
    return mat


def build_encoding_unitary(features: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Product of RY(π * f_i) gates on each qubit.
    features: 1-D array of length >= n_qubits, values in [-1, 1].
    """
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=np.float64)
    for i in range(n_qubits):
        U = embed_single(RY(np.pi * float(features[i])), i, n_qubits) @ U
    return U


def build_reservoir_unitary(ry_params: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Fixed random reservoir: alternating RY layers + CNOT ring.
    ry_params: (n_layers, n_qubits)
    """
    dim = 2 ** n_qubits
    U = np.eye(dim, dtype=np.float64)
    n_layers = ry_params.shape[0]
    for layer in range(n_layers):
        for i in range(n_qubits):
            U = embed_single(RY(ry_params[layer, i]), i, n_qubits) @ U
        for i in range(n_qubits - 1):
            U = embed_cnot(i, i + 1, n_qubits) @ U
        U = embed_cnot(n_qubits - 1, 0, n_qubits) @ U  # ring closure
    return U


def z_observable(qubit: int, n_qubits: int) -> np.ndarray:
    """⟨Z_i⟩ observable in full Hilbert space."""
    return embed_single(Z, qubit, n_qubits)


def zero_state(n_qubits: int) -> np.ndarray:
    """Column vector |0...0⟩ as shape (2^n, 1)."""
    v = np.zeros((2 ** n_qubits, 1), dtype=np.float64)
    v[0, 0] = 1.0
    return v
