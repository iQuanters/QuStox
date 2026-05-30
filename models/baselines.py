"""
Classical baselines for honest comparison against the quantum reservoir.

  * ClassicalSVR : the SAME SVR readout, but trained on the raw window of past
                   returns instead of the quantum features. Isolates whatever
                   the quantum reservoir adds on top of the identical input.
  * naive_persistence : predict next return = last return (a momentum null model).
"""

import numpy as np

from .readout import SVRReadout


class ClassicalSVR(SVRReadout):
    """Identical SVR, applied directly to the raw lag-window features."""
    pass


def naive_persistence(enc_window_test: np.ndarray) -> np.ndarray:
    """Predict next return = most recent encoded return in the window."""
    return enc_window_test[:, -1]
