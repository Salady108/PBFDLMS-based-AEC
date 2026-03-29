from __future__ import annotations

import numpy as np


def snr_db(clean: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    p_sig = float(np.mean(clean * clean))
    err = clean - test
    p_err = float(np.mean(err * err))
    return float(10.0 * np.log10((p_sig + eps) / (p_err + eps)))
