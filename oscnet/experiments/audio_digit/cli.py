"""CLI for the audio-digit resonator filter-bank (RFB) frontend probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import AudioDigitConfig
from .runner import run_audio_digit_experiment, run_stage0a_sweep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Spoken-digit RFB frontend probe "
            "(resonator / mel / stft / raw / equal-frequency / HORN head)."
        )
    )
    p.add_argument(
        "--frontend",
        default="resonator",
        choices=(
            "resonator",
            "resonator_learn",
            "resonator_equal",
            "mel",
            "stft",
            "raw",
            "raw_wide",
            "conv1d",
        ),
    )
    p.add_argument("--sweep", action="store_true", help="Run frontend attribution sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-bands", type=int, default=16)
    p.add_argument("--quality-factor", type=float, default=4.0)
    p.add_argument("--pool", default="log_rms", choices=("log_rms", "rms", "mean", "max"))
    p.add_argument(
        "--feature-mode",
        default="pooled",
        choices=("pooled", "frames"),
    )
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument(
        "--learnable-frontend",
        action="store_true",
        help="Stage 2: learn resonator {ω, Q}",
    )
    p.add_argument("--head-kind", default="mlp", choices=("mlp", "horn", "gru"))
    p.add_argument("--head-hidden-dim", type=int, default=32)
    p.add_argument("--horn-steps", type=int, default=8)
    p.add_argument(
        "--horn-coupling-kind",
        default="fractal_fixed",
        choices=("fractal_fixed", "dense", "dense_tonotopic"),
    )
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=500)
    p.add_argument("--sample-rate", type=float, default=8000.0)
    p.add_argument("--duration-sec", type=float, default=0.5)
    p.add_argument(
        "--data-source",
        default="synthetic",
        choices=("synthetic", "speech_commands"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/audio_digit"))
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    return p


def config_from_args(args: argparse.Namespace) -> AudioDigitConfig:
    return AudioDigitConfig(
        frontend=args.frontend,
        num_bands=args.num_bands,
        quality_factor=args.quality_factor,
        pool=args.pool,
        feature_mode=args.feature_mode,
        num_frames=args.num_frames,
        learnable_frontend=bool(args.learnable_frontend)
        or args.frontend == "resonator_learn",
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        head_kind=args.head_kind,
        head_hidden_dim=args.head_hidden_dim,
        horn_steps=args.horn_steps,
        horn_coupling_kind=args.horn_coupling_kind,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        data_source=args.data_source,
        output_dir=Path(args.output_dir),
    )


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sweep:
        base = config_from_args(args)
        summary = run_stage0a_sweep(
            seeds=tuple(args.seeds),
            base=base,
        )
        print(f"wrote {summary['output_path']}")
        for name, stats in summary["by_frontend"].items():
            print(
                f"  {name:16s}  acc={stats['mean_eval_accuracy']:.4f} "
                f"± {stats['std_eval_accuracy']:.4f}  (n={stats['n']})"
            )
        return
    result = run_audio_digit_experiment(config_from_args(args))
    print(
        f"{result['run_tag']} "
        f"acc={result['final_eval_accuracy']:.4f} "
        f"params={result['trainable_params']} "
        f"-> {result['output_path']}"
    )


if __name__ == "__main__":
    main()
