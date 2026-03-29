from __future__ import annotations

import numpy as np


def plot_waveforms(
    clean: np.ndarray,
    noisy: np.ndarray,
    recovered: np.ndarray,
    fs: int,
    output_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    t = np.arange(clean.size, dtype=np.float64) / float(fs)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, clean, color="tab:green", linewidth=0.8)
    axes[0].set_title("Clean Waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, noisy, color="tab:red", linewidth=0.8)
    axes[1].set_title("Noisy + Echo Waveform")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, recovered, color="tab:blue", linewidth=0.8)
    axes[2].set_title("Recovered Waveform")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
