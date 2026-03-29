from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def resample_linear(x: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    if src_fs <= 0 or dst_fs <= 0:
        raise ValueError("sample rates must be > 0")
    if src_fs == dst_fs:
        return x.astype(np.float64, copy=True)
    if x.size < 2:
        return x.astype(np.float64, copy=True)
    n_out = max(1, int(round((len(x) * dst_fs) / src_fs)))
    xp = np.arange(len(x), dtype=np.float64)
    xq = np.linspace(0.0, float(len(x) - 1), n_out, dtype=np.float64)
    return np.interp(xq, xp, x).astype(np.float64)


def load_audio_mono(path: str | Path, target_fs: int) -> tuple[np.ndarray, int]:
    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(str(path))
        sample_width_bytes = int(seg.sample_width)
        channels = int(seg.channels)
        src_fs = int(seg.frame_rate)

        samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        den = float(1 << (8 * sample_width_bytes - 1))
        x = samples / max(den, 1.0)
        x = np.clip(x, -1.0, 1.0)

        if src_fs != target_fs:
            x = resample_linear(x, src_fs, target_fs)

        return x.astype(np.float64, copy=False), target_fs
    except Exception:
        pass

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(target_fs),
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg decode failed: {err}")

        x = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)
        if x.size == 0:
            raise RuntimeError("Decoded audio is empty")

        x = np.clip(x, -1.0, 1.0)
        return x, target_fs
    except Exception as exc:
        raise RuntimeError(
            "Could not decode input audio. Install requirements (including imageio-ffmpeg), "
            "or install system ffmpeg/ffprobe."
        ) from exc


def write_wav_pcm16(path: str | Path, x: np.ndarray, fs: int) -> None:
    import wave

    x16 = np.clip(x, -1.0, 1.0)
    x16 = np.round(x16 * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(x16.tobytes())
