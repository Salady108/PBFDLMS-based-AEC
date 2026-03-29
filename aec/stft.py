from __future__ import annotations

import numpy as np


def stft_hann(
    x: np.ndarray,
    *,
    frame_len: int = 320,
    hop: int = 160,
    center: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be > 0")
    if hop > frame_len:
        raise ValueError("hop must be <= frame_len")

    win = np.hanning(frame_len + 1)[:-1].astype(np.float64)
    win_energy = float(np.sum(win * win))
    if win_energy < eps:
        raise ValueError("Degenerate STFT window")

    if center:
        pad = frame_len // 2
        xp = np.pad(x.astype(np.float64, copy=False), (pad, pad))
    else:
        xp = x.astype(np.float64, copy=False)

    if len(xp) < frame_len:
        xp = np.pad(xp, (0, frame_len - len(xp)))

    n_frames = 1 + (len(xp) - frame_len) // hop
    out = np.empty((n_frames, frame_len // 2 + 1), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        frame = xp[start : start + frame_len]
        out[i] = np.fft.rfft(frame * win)
    return out
