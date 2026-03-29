"""Stage-1 acoustic echo cancellation: time alignment + PBFDLMS.

Modular entrypoint for:
- synthetic AEC demo
- real-audio restoration demo
"""

from __future__ import annotations

from aec import build_arg_parser, run_real_audio_restoration, run_synthetic_demo


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.demo and not args.real_audio:
        args.demo = True

    if args.real_audio:
        plot_path = args.plot_path
        if plot_path == "waveforms.png":
            plot_path = f"{args.output_prefix}_waveforms.png"
        run_real_audio_restoration(
            input_audio=args.input_audio,
            fs=args.fs,
            seconds=args.seconds,
            block_len=args.block_len,
            filter_len=args.filter_len,
            step_size=args.mu,
            seed=args.seed,
            echo_delay_ms=args.echo_delay_ms,
            echo_decay=args.echo_decay,
            echo_gain=args.echo_gain,
            noise_db=args.noise_db,
            max_delay_ms=args.max_delay_ms,
            delay_est_seconds=args.delay_est_seconds,
            beta=args.beta,
            output_prefix=args.output_prefix,
            plot_path=plot_path,
        )
    else:
        run_synthetic_demo(
            fs=args.fs,
            seconds=args.seconds,
            block_len=args.block_len,
            filter_len=args.filter_len,
            step_size=args.mu,
            seed=args.seed,
            noise_db=args.noise_db,
            true_delay_ms=args.true_delay_ms,
            max_delay_ms=args.max_delay_ms,
            delay_est_seconds=args.delay_est_seconds,
            beta=args.beta,
        )


if __name__ == "__main__":
    main()
