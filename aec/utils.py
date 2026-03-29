from __future__ import annotations

import numpy as np


def next_multiple(value: int, base: int) -> int:
    return ((value + base - 1) // base) * base


def next_pow2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length())


def db10(x: np.ndarray | float, *, floor: float = 1e-12) -> np.ndarray | float:
    x_safe = np.maximum(x, floor)
    return 10.0 * np.log10(x_safe)


def colored_noise(rng: np.random.Generator, n: int, a: float = 0.95) -> np.ndarray:
    w = rng.normal(size=n)
    y = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        s = a * s + w[i]
        y[i] = s
    y *= np.sqrt(1.0 - a * a)
    return y


def apply_delay(x: np.ndarray, delay_samples: int) -> np.ndarray:
    if delay_samples <= 0:
        return x.copy()
    return np.pad(x, (delay_samples, 0))[: len(x)]
