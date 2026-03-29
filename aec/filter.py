from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .utils import next_multiple


@dataclass
class TimeAlignedPBFDAF:
    filter_len: int
    block_len: int = 160
    step_size: float = 0.5
    reg: float = 1e-6
    power_smooth: float = 0.8
    msc_smooth: float = 0.9
    error_clip_beta: float = 2.0
    min_x_power: float = 1e-10

    def __post_init__(self) -> None:
        if self.block_len <= 0:
            raise ValueError("block_len must be > 0")
        if self.filter_len <= 0:
            raise ValueError("filter_len must be > 0")
        if not (0.0 < self.step_size):
            raise ValueError("step_size must be > 0")
        if self.reg <= 0.0:
            raise ValueError("reg must be > 0")
        if not (0.0 <= self.power_smooth < 1.0):
            raise ValueError("power_smooth must be in [0, 1)")
        if not (0.0 <= self.msc_smooth < 1.0):
            raise ValueError("msc_smooth must be in [0, 1)")
        if self.error_clip_beta < 1.0:
            raise ValueError("error_clip_beta must be >= 1")

        self.fft_len = 2 * self.block_len
        self.filter_len = next_multiple(self.filter_len, self.block_len)
        self.n_partitions = self.filter_len // self.block_len

        self._W = np.zeros((self.n_partitions, self.fft_len), dtype=np.complex128)
        self._X = np.zeros((self.n_partitions, self.fft_len), dtype=np.complex128)
        self._x_overlap = np.zeros(self.block_len, dtype=np.float64)
        self._d_overlap = np.zeros(self.block_len, dtype=np.float64)
        self._power = np.full(self.fft_len, 1e-6, dtype=np.float64)
        self._P_xx = np.full(self.fft_len, 1e-6, dtype=np.float64)
        self._P_dd = np.full(self.fft_len, 1e-6, dtype=np.float64)
        self._P_dx = np.zeros(self.fft_len, dtype=np.complex128)

    def reset(self) -> None:
        self._W.fill(0.0)
        self._X.fill(0.0)
        self._x_overlap.fill(0.0)
        self._d_overlap.fill(0.0)
        self._power.fill(1e-6)
        self._P_xx.fill(1e-6)
        self._P_dd.fill(1e-6)
        self._P_dx.fill(0.0)

    def get_filter_time_domain(self) -> np.ndarray:
        w_parts = np.fft.ifft(self._W, axis=1).real
        w_parts[:, self.block_len :] = 0.0
        w = w_parts[:, : self.block_len].reshape(-1)
        return w.copy()

    def process_block(
        self, x_block: np.ndarray, d_block: np.ndarray, *, adapt: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if x_block.shape != (self.block_len,) or d_block.shape != (self.block_len,):
            raise ValueError(
                f"Blocks must have shape ({self.block_len},), got {x_block.shape} and {d_block.shape}"
            )

        x_power = float(np.mean(x_block * x_block))
        if x_power < self.min_x_power:
            adapt = False

        x_ext = np.concatenate([self._x_overlap, x_block])
        d_ext = np.concatenate([self._d_overlap, d_block])
        self._x_overlap = x_block.astype(np.float64, copy=True)
        self._d_overlap = d_block.astype(np.float64, copy=True)

        x_fft = np.fft.fft(x_ext)
        d_fft = np.fft.fft(d_ext)

        self._P_xx = self.msc_smooth * self._P_xx + (1.0 - self.msc_smooth) * (
            np.abs(x_fft) ** 2
        )
        self._P_dd = self.msc_smooth * self._P_dd + (1.0 - self.msc_smooth) * (
            np.abs(d_fft) ** 2
        )
        self._P_dx = self.msc_smooth * self._P_dx + (1.0 - self.msc_smooth) * (
            d_fft * np.conj(x_fft)
        )
        msc = (np.abs(self._P_dx) ** 2) / (self._P_dd * self._P_xx + 1e-12)
        msc = np.clip(msc.real, 0.0, 1.0)
        mu_vec = self.step_size * msc
        if not adapt:
            mu_vec = np.zeros_like(mu_vec)

        if self.n_partitions > 1:
            self._X[1:] = self._X[:-1]
        self._X[0] = x_fft

        y_fft = np.sum(self._W * self._X, axis=0)
        e_fft = d_fft - y_fft

        if self.error_clip_beta is not None:
            e_mag = np.abs(e_fft)
            d_mag = np.abs(d_fft)
            limit = self.error_clip_beta * d_mag
            scale = np.minimum(1.0, limit / (e_mag + 1e-12))
            e_fft = e_fft * scale

        e_ext = np.fft.ifft(e_fft).real
        e_block = e_ext[self.block_len :]
        y_block = d_block - e_block

        if adapt:
            e_ta = np.concatenate([np.zeros(self.block_len, dtype=np.float64), e_block])
            e_ta_fft = np.fft.fft(e_ta)

            power_inst = np.sum(np.abs(self._X) ** 2, axis=0).real
            self._power = self.power_smooth * self._power + (1.0 - self.power_smooth) * power_inst
            denom = self._power + self.reg

            self._W += (
                mu_vec[np.newaxis, :]
                * (np.conj(self._X) * e_ta_fft[np.newaxis, :])
                / denom[np.newaxis, :]
            )

            w_time = np.fft.ifft(self._W, axis=1).real
            w_time[:, self.block_len :] = 0.0
            self._W = np.fft.fft(w_time, axis=1)

        return e_block, y_block
