"""
Quantum Reservoir Computer (XpyQ backend).

Uses the hackathon-provided XpyQ linear-algebra API for all matrix operations.
Gate matrices are constructed locally in numpy (constant overhead), then the
dynamic operations — encoding the market state and propagating through the
reservoir — are dispatched to XpyQ via matmul calls.

Why XpyQ for the matmul?
  The XpyQ API represents the 'quantum compute' layer of this hybrid system.
  We offload the linear-algebra core (state evolution, inner products) to
  the hosted service, which can in principle run on quantum-accelerated
  hardware.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Allow importing la_client from the SDK folder
_SDK_DIR = Path(__file__).parent.parent / "Drops" / "internal_config_sdk"
if str(_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(_SDK_DIR))

from la_client import LinearAlgebraClient, LinearAlgebraAPIError
from .gates import (
    build_encoding_unitary,
    build_reservoir_unitary,
    z_observable,
    zero_state,
)


def _to_nested_list(arr: np.ndarray):
    return arr.tolist()


class XpyQReservoir:
    """
    XpyQ-backed fixed quantum reservoir.

    Parameters
    ----------
    api_key  : XpyQ API key (or set XPYQ_API_KEY env var).
    n_qubits : Number of qubits (recommend 4 — 16×16 state space).
    n_layers : Reservoir depth.
    seed     : RNG seed for fixed reservoir weights.
    """

    def __init__(
        self,
        api_key: str | None = None,
        n_qubits: int = 4,
        n_layers: int = 3,
        seed: int = 42,
    ):
        key = api_key or os.environ.get("XPYQ_API_KEY")
        if not key:
            raise ValueError(
                "XpyQ API key required. Pass api_key= or set XPYQ_API_KEY env var."
            )
        self.client = LinearAlgebraClient(api_key=key)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)
        ry_params = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits))

        # Precompute fixed reservoir unitary (numpy, done once)
        self._U_res = build_reservoir_unitary(ry_params, n_qubits)

        # Precompute Z observables
        self._Z_obs = [z_observable(i, n_qubits) for i in range(n_qubits)]

        # Initial state |0...0⟩ as (2^n, 1) list
        self._zero = _to_nested_list(zero_state(n_qubits))

    @property
    def output_dim(self) -> int:
        return self.n_qubits

    def _circuit(self, features: np.ndarray) -> list[float]:
        """Run one sample through the quantum circuit using XpyQ matmul."""
        # 1. Build encoding unitary (per-sample, features-dependent)
        U_enc = build_encoding_unitary(features[: self.n_qubits], self.n_qubits)

        # 2. Full circuit unitary = U_reservoir @ U_encoding (numpy, fast)
        U_total = self._U_res @ U_enc

        # 3. Apply circuit to |0...0⟩ via XpyQ  (the quantum compute step)
        state = self.client.matmul(
            _to_nested_list(U_total), self._zero
        )  # shape (2^n, 1)

        # 4. Compute ⟨Zᵢ⟩ = ψᵀ Zᵢ ψ  via two XpyQ matmul calls per qubit
        state_T = [[row[0] for row in state]]  # (1, 2^n)
        exp_vals = []
        for Z_i in self._Z_obs:
            Zs = self.client.matmul(_to_nested_list(Z_i), state)  # (2^n, 1)
            ev = self.client.matmul(state_T, Zs)                  # (1, 1)
            exp_vals.append(float(ev[0][0]))

        return exp_vals

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n_samples, n_features), values in [-1, 1].

        Returns
        -------
        Q : (n_samples, n_qubits) array of ⟨Zᵢ⟩ values in [-1, 1].
        """
        results = []
        for i, x in enumerate(X):
            try:
                vals = self._circuit(np.asarray(x, dtype=float))
                results.append(vals)
            except LinearAlgebraAPIError as e:
                print(f"[xpyq] API error on sample {i}: {e}")
                results.append([0.0] * self.n_qubits)
        return np.array(results, dtype=np.float64)
