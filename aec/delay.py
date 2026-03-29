from __future__ import annotations

import numpy as np

from .utils import next_pow2


def estimate_delay_gcc_phat(
    reference: np.ndarray,
    mic: np.ndarray,
    *,
    fs: int,
    max_delay_ms: float = 700.0,
    seconds: float = 1.0,
    eps: float = 1e-12,
) -> int:
    if reference.ndim != 1 or mic.ndim != 1:
        raise ValueError("reference and mic must be 1-D arrays")
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if max_delay_ms <= 0:
        raise ValueError("max_delay_ms must be > 0")
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    max_shift = int(round((max_delay_ms / 1000.0) * fs))
    if max_shift < 1:
        return 0

    n_seg = int(round(seconds * fs))
    n_seg = min(n_seg, len(reference), len(mic))
    if n_seg <= 2:
        return 0

    ref = reference[:n_seg].astype(np.float64, copy=False)
    sig = mic[:n_seg].astype(np.float64, copy=False)

    ref = ref - float(np.mean(ref))
    sig = sig - float(np.mean(sig))

    nfft = next_pow2(len(ref) + len(sig))
    ref_fft = np.fft.fft(ref, n=nfft)
    sig_fft = np.fft.fft(sig, n=nfft)
    g = sig_fft * np.conj(ref_fft)
    g /= np.abs(g) + eps
    cc = np.fft.ifft(g).real

    cc = np.concatenate([cc[-max_shift:], cc[: max_shift + 1]])
    shift = int(np.argmax(cc) - max_shift)
    if shift < 0:
        shift = 0
    return shift
