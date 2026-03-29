from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Time-aligned partitioned block frequency-domain adaptive filtering (PBFDAF) demo"
    )
    p.add_argument("--demo", action="store_true", help="Run synthetic AEC demo")
    p.add_argument(
        "--real-audio",
        action="store_true",
        help="Run restoration on real audio (loads AAC, creates noisy-echo version, recovers clean)",
    )
    p.add_argument(
        "--input-audio",
        type=str,
        default="Myvoice.aac",
        help="Path to clean real audio input (AAC recommended)",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default="myvoice",
        help="Prefix for generated WAV files",
    )
    p.add_argument(
        "--plot-path",
        type=str,
        default="waveforms.png",
        help="Path for the generated clean/noisy/recovered waveform plot",
    )
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument(
        "--block-len",
        type=int,
        default=160,
        help="Hop / block length N (FFT length is 2N; paper uses N=160 at 16 kHz)",
    )
    p.add_argument(
        "--filter-len",
        type=int,
        default=3840,
        help="Adaptive filter length in samples (paper uses 24 sub-blocks => 24*N)",
    )
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Error magnitude constraint factor beta in |E|<=beta|D| (paper uses beta=2)",
    )
    p.add_argument(
        "--true-delay-ms",
        type=float,
        default=350.0,
        help="(demo) Inject a true delay between reference and mic (ms)",
    )
    p.add_argument(
        "--max-delay-ms",
        type=float,
        default=700.0,
        help="Maximum delay searched by GCC-PHAT (ms)",
    )
    p.add_argument(
        "--delay-est-seconds",
        type=float,
        default=1.0,
        help="Seconds used from start for GCC-PHAT delay estimation",
    )
    p.add_argument(
        "--noise-db",
        type=float,
        default=-30.0,
        help="Additive noise level relative to clean RMS (dB)",
    )
    p.add_argument(
        "--echo-delay-ms",
        type=float,
        default=180.0,
        help="(real-audio) Delay used to synthesize echo (ms)",
    )
    p.add_argument(
        "--echo-decay",
        type=float,
        default=0.12,
        help="(real-audio) Exponential decay factor controlling echo tail",
    )
    p.add_argument(
        "--echo-gain",
        type=float,
        default=0.6,
        help="(real-audio) Echo path gain",
    )
    return p
