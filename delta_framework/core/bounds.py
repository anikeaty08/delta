"""Simple theoretical bound helpers."""

from __future__ import annotations

import math
from typing import Dict


def pac_bound(
    *,
    N_old: int,
    N_new: int,
    delta: float,
    correction_term: float = 0.0,
) -> Dict[str, float]:
    """Compute a basic PAC-style bound on an accuracy gap.

    Formula:
      epsilon = sqrt( (1/(2*N_new)) * log(2/delta) ) + correction_term
    """
    _ = N_old  # included for signature parity / future extensions

    N_new_i = int(N_new)
    delta_f = float(delta)
    correction = float(correction_term)

    if N_new_i <= 0:
        eps = 1.0 + correction
    else:
        delta_f = min(max(delta_f, 1e-12), 0.999999)
        eps = math.sqrt((1.0 / (2.0 * N_new_i)) * math.log(2.0 / delta_f)) + correction

    return {
        "guaranteed_epsilon": float(eps),
        "confidence_level": float(1.0 - delta_f),
    }


