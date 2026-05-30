"""
Realistic (but gentle) noise model for the Qiskit reservoir.

The goal is to validate that the reservoir dynamics survive on near-term
hardware — NOT to drown them. Levels sit at the optimistic end of current
superconducting devices:

  * 1-qubit gate depolarizing error : 1e-3
  * 2-qubit gate depolarizing error : 1e-2
  * readout (measurement) error     : 1e-2
  * thermal relaxation              : T1=120 us, T2=90 us, gate times ~ 50/300 ns

Errors are attached to the reservoir's NATIVE gates (rx/ry for 1-qubit;
rxx/rzz for the Ising couplings). This keeps the simulated circuit shallow —
decomposing every coupling into cx chains would multiply the gate count and
make the (already per-shot-trajectory) noisy simulation far slower, without
changing the qualitative dynamics. rz is treated as a virtual, noiseless gate,
as on real hardware.
"""

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)

# Native gates the reservoir is simulated with (no cx decomposition).
BASIS_GATES = ["rx", "ry", "rz", "rxx", "rzz"]


def build_noise_model(
    p1: float = 1e-3,
    p2: float = 1e-2,
    p_readout: float = 1e-2,
    t1_us: float = 120.0,
    t2_us: float = 90.0,
    t_1q_ns: float = 50.0,
    t_2q_ns: float = 300.0,
) -> NoiseModel:
    """Construct a gentle, realistic superconducting-style noise model."""
    nm = NoiseModel(basis_gates=BASIS_GATES)

    t1, t2 = t1_us * 1e3, t2_us * 1e3  # ns

    # --- single-qubit rotations: depolarizing ∘ thermal relaxation -----------
    relax_1q = thermal_relaxation_error(t1, t2, t_1q_ns)
    err_1q = depolarizing_error(p1, 1).compose(relax_1q)
    nm.add_all_qubit_quantum_error(err_1q, ["rx", "ry"])

    # --- two-qubit Ising couplings: depolarizing ∘ (relax ⊗ relax) -----------
    relax_2q = thermal_relaxation_error(t1, t2, t_2q_ns).tensor(
        thermal_relaxation_error(t1, t2, t_2q_ns)
    )
    err_2q = depolarizing_error(p2, 2).compose(relax_2q)
    nm.add_all_qubit_quantum_error(err_2q, ["rxx", "rzz"])

    # --- measurement (readout) error -----------------------------------------
    ro = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
    nm.add_all_qubit_readout_error(ro)

    return nm
