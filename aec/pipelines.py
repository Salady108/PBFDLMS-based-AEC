from __future__ import annotations

import numpy as np

from .audio import load_audio_mono, write_wav_pcm16
from .delay import estimate_delay_gcc_phat
from .filter import TimeAlignedPBFDAF
from .metrics import snr_db
from .plotting import plot_waveforms
from .stft import stft_hann
from .utils import apply_delay, colored_noise, db10


def run_synthetic_demo(
    *,
    fs: int,
    seconds: float,
    block_len: int,
    filter_len: int,
    step_size: float,
    seed: int,
    noise_db: float,
    true_delay_ms: float,
    max_delay_ms: float,
    delay_est_seconds: float,
    beta: float,
) -> None:
    rng = np.random.default_rng(seed)
    n = int(round(fs * seconds))

    x = colored_noise(rng, n, a=0.97)
    x /= np.max(np.abs(x)) + 1e-12

    h = rng.normal(size=filter_len)
    decay = np.exp(-np.arange(filter_len) / (0.12 * filter_len))
    h *= decay
    h /= np.linalg.norm(h) + 1e-12
    h *= 0.8

    delay_samples = int(round((true_delay_ms / 1000.0) * fs))
    x_delayed = apply_delay(x, delay_samples)
    echo = np.convolve(x_delayed, h, mode="full")[:n]

    noise = rng.normal(size=n)
    noise *= np.sqrt(np.mean(echo * echo) + 1e-12) * (10.0 ** (noise_db / 20.0))
    d = echo + noise

    delay_hat = estimate_delay_gcc_phat(
        x,
        d,
        fs=fs,
        max_delay_ms=max_delay_ms,
        seconds=delay_est_seconds,
    )
    delay_align = (delay_hat // block_len) * block_len
    x_aligned = apply_delay(x, delay_align)

    aec = TimeAlignedPBFDAF(
        filter_len=filter_len,
        block_len=block_len,
        step_size=step_size,
        reg=1e-6,
        power_smooth=0.8,
        error_clip_beta=beta,
        msc_smooth=0.9,
    )

    n_blocks = int(np.ceil(n / block_len))
    x_pad = np.pad(x_aligned, (0, n_blocks * block_len - n))
    d_pad = np.pad(d, (0, n_blocks * block_len - n))

    e_pad = np.zeros_like(d_pad)

    for b in range(n_blocks):
        i0 = b * block_len
        i1 = i0 + block_len
        e_b, _ = aec.process_block(x_pad[i0:i1], d_pad[i0:i1], adapt=True)
        e_pad[i0:i1] = e_b

    e = e_pad[:n]

    frame_len = 2 * block_len
    hop = block_len
    r_lk = stft_hann(x_aligned, frame_len=frame_len, hop=hop, center=False)
    d_lk = stft_hann(d, frame_len=frame_len, hop=hop, center=False)
    e_lk = stft_hann(e, frame_len=frame_len, hop=hop, center=False)

    alpha = 0.99
    pd = 0.0
    pe = 0.0
    erle = np.empty(n, dtype=np.float64)
    for i in range(n):
        pd = alpha * pd + (1.0 - alpha) * float(d[i] * d[i])
        pe = alpha * pe + (1.0 - alpha) * float(e[i] * e[i])
        erle[i] = db10((pd + 1e-12) / (pe + 1e-12))

    tail = erle[int(0.8 * n) :]
    erle_final = float(np.median(tail))
    w_est = aec.get_filter_time_domain()[:filter_len]
    misalign = 20.0 * np.log10(
        (np.linalg.norm(h - w_est) + 1e-12) / (np.linalg.norm(h) + 1e-12)
    )

    print("Synthetic AEC demo")
    print(f"  fs={fs} Hz, duration={seconds:.2f} s")
    print(f"  block_len={block_len}, filter_len={filter_len}")
    print(f"  step_size(mu)={step_size}")
    print(f"  true delay = {delay_samples} samples ({true_delay_ms:.1f} ms)")
    print(f"  estimated delay = {delay_hat} samples")
    print(f"  applied delay (quantized) = {delay_align} samples")
    print(f"  beta(error clip)={beta}")
    print(f"  STFT (Hann) frames={r_lk.shape[0]}, bins={r_lk.shape[1]}")
    print(f"  final ERLE (median over last 20%) = {erle_final:.2f} dB")
    print(f"  final misalignment = {misalign:.2f} dB")


def run_real_audio_restoration(
    *,
    input_audio: str,
    fs: int,
    seconds: float,
    block_len: int,
    filter_len: int,
    step_size: float,
    seed: int,
    echo_delay_ms: float,
    echo_decay: float,
    echo_gain: float,
    noise_db: float,
    max_delay_ms: float,
    delay_est_seconds: float,
    beta: float,
    output_prefix: str,
    plot_path: str,
) -> None:
    rng = np.random.default_rng(seed)
    clean_full, fs = load_audio_mono(input_audio, fs)
    if clean_full.size < 2 * block_len:
        raise ValueError("Input audio too short for block processing")

    if seconds > 0:
        n = min(int(round(seconds * fs)), clean_full.size)
    else:
        n = clean_full.size
    clean = clean_full[:n].astype(np.float64, copy=False)
    clean /= np.max(np.abs(clean)) + 1e-12

    echo_taps = max(block_len, filter_len // 2)
    h = rng.normal(size=echo_taps)
    decay = np.exp(-np.arange(echo_taps) / (max(echo_decay, 1e-3) * echo_taps))
    h *= decay
    h /= np.linalg.norm(h) + 1e-12
    h *= echo_gain

    delay_samples = int(round((echo_delay_ms / 1000.0) * fs))
    echo = np.convolve(apply_delay(clean, delay_samples), h, mode="full")[:n]

    noise = rng.normal(size=n)
    clean_rms = np.sqrt(np.mean(clean * clean) + 1e-12)
    noise_rms = clean_rms * (10.0 ** (noise_db / 20.0))
    noise *= noise_rms / (np.sqrt(np.mean(noise * noise)) + 1e-12)

    noisy_echo = clean + echo + noise
    noisy_echo /= np.max(np.abs(noisy_echo)) + 1e-12

    delay_hat = estimate_delay_gcc_phat(
        reference=clean,
        mic=noisy_echo,
        fs=fs,
        max_delay_ms=max_delay_ms,
        seconds=delay_est_seconds,
    )
    delay_align = (delay_hat // block_len) * block_len
    x_aligned = apply_delay(clean, delay_align)

    model = TimeAlignedPBFDAF(
        filter_len=filter_len,
        block_len=block_len,
        step_size=step_size,
        reg=1e-6,
        power_smooth=0.8,
        error_clip_beta=beta,
        msc_smooth=0.9,
    )

    n_blocks = int(np.ceil(n / block_len))
    x_pad = np.pad(x_aligned, (0, n_blocks * block_len - n))
    d_pad = np.pad(noisy_echo, (0, n_blocks * block_len - n))

    y_pad = np.zeros_like(d_pad)

    for b in range(n_blocks):
        i0 = b * block_len
        i1 = i0 + block_len
        _, y_b = model.process_block(x_pad[i0:i1], d_pad[i0:i1], adapt=True)
        y_pad[i0:i1] = y_b

    recovered = y_pad[:n]
    recovered /= np.max(np.abs(recovered)) + 1e-12

    out_noisy = f"{output_prefix}_noisy_echo.wav"
    out_recovered = f"{output_prefix}_recovered_clean.wav"
    out_clean = f"{output_prefix}_clean_reference.wav"

    write_wav_pcm16(out_clean, clean, fs)
    write_wav_pcm16(out_noisy, noisy_echo, fs)
    write_wav_pcm16(out_recovered, recovered, fs)
    plot_waveforms(clean, noisy_echo, recovered, fs, plot_path)

    snr_in = snr_db(clean, noisy_echo)
    snr_out = snr_db(clean, recovered)

    print("Real-audio restoration demo")
    print(f"  input file={input_audio}")
    print(f"  fs={fs} Hz, used duration={n / fs:.2f} s")
    print(f"  block_len={block_len}, filter_len={filter_len}, step_size(mu)={step_size}")
    print(f"  injected echo delay={delay_samples} samples ({echo_delay_ms:.1f} ms)")
    print(f"  estimated delay={delay_hat} samples, applied={delay_align} samples")
    print(f"  input SNR(clean vs noisy_echo)={snr_in:.2f} dB")
    print(f"  output SNR(clean vs recovered)={snr_out:.2f} dB")
    print(f"  wrote: {out_clean}")
    print(f"  wrote: {out_noisy}")
    print(f"  wrote: {out_recovered}")
    print(f"  wrote: {plot_path}")
