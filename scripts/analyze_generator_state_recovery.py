"""Noise-then-settle recovery probe for trained image generators.

This eval-only diagnostic asks whether a trained model's settling dynamics can
*repair* a corrupted detail-bearing state. It fits per-image final states
through the frozen decoder, perturbs those states at several relative noise
scales, settles for a range of depths, and reports paired reconstruction
error against both the un-settled noisy baseline and the clean fit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from oscnet.analysis.generator_checkpoints import load_generator_checkpoint_model
from oscnet.analysis.generator_state_prior import save_state_prior_contact_sheet
from oscnet.experiments.harness import write_json
from oscnet.experiments.mnist_autoencoder import load_mnist_data
from oscnet.experiments.mnist_generator import (
    compute_generator_state_fitting_probe,
)


DEFAULT_PRESET = (
    "sparse_horn_cifar10_rgb_current_multimode2_"
    "retinotopic_anchor030_prior_class_patch005"
)


def _parse_float_list(text: str) -> List[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def _condition_rows(
    probe: Dict[str, Any],
    *,
    noise_scales: Sequence[float],
    recovery_settle_steps: Sequence[int],
    clean_settle_steps: Sequence[int],
    occlusion_fractions: Sequence[float] = (),
) -> List[Dict[str, Any]]:
    """Flatten probe metrics into one tidy row per (condition, settle depth)."""

    def collect(prefix: str) -> Dict[str, float]:
        marker = f"{prefix}_"
        return {
            key[len(marker):]: value
            for key, value in probe.items()
            if key.startswith(marker) and isinstance(value, (int, float))
        }

    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "condition": "clean",
            "noise_scale": 0.0,
            "settle_steps": 0,
            **collect("fit"),
        }
    )
    for step in clean_settle_steps:
        step = int(step)
        if step <= 0:
            continue
        clean_row = collect(f"settle_{step:03d}")
        if clean_row:
            rows.append(
                {
                    "condition": "clean",
                    "noise_scale": 0.0,
                    "settle_steps": step,
                    **clean_row,
                }
            )
        control_row = collect(f"noise_{step:03d}")
        if control_row:
            rows.append(
                {
                    "condition": "displacement_matched_noise",
                    "noise_scale": float("nan"),
                    "settle_steps": step,
                    **control_row,
                }
            )
    for scale_index, noise_scale in enumerate(noise_scales):
        prefix = f"recover_n{scale_index}"
        for step in (0, *recovery_settle_steps):
            step = int(step)
            step_row = collect(f"{prefix}_settle_{step:03d}")
            if not step_row:
                continue
            rows.append(
                {
                    "condition": "noise_then_settle",
                    "noise_scale": float(noise_scale),
                    "settle_steps": step,
                    "state_displacement_rms": probe.get(
                        f"{prefix}_state_displacement_rms"
                    ),
                    **step_row,
                }
            )
    for fraction_index, fraction in enumerate(occlusion_fractions):
        prefix = f"occl_f{fraction_index}"
        for step in (0, *recovery_settle_steps):
            step = int(step)
            step_row = collect(f"{prefix}_settle_{step:03d}")
            if not step_row:
                continue
            rows.append(
                {
                    "condition": "occlude_then_settle",
                    "noise_scale": float(fraction),
                    "settle_steps": step,
                    "patch_sites": probe.get(f"{prefix}_patch_sites"),
                    **step_row,
                }
            )
    return rows


def write_recovery_rows_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = ["condition", "noise_scale", "settle_steps"]
    keys = sorted({key for row in rows for key in row} - set(lead))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*lead, *keys])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit final states to real images through a frozen decoder, "
            "perturb them, and test whether settling repairs the decode."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--fit-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--noise-scales", default="0.125,0.25,0.5,1.0")
    parser.add_argument("--occlusion-fractions", default="0.0625,0.125,0.25")
    parser.add_argument("--recovery-settle-steps", default="1,2,4,8,16")
    parser.add_argument("--clean-settle-steps", default="0,1,2,4,8,16")
    parser.add_argument("--probe-seed", type=int, default=9203)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument(
        "--output-prefix",
        default="generator_state_recovery_probe",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    noise_scales = _parse_float_list(args.noise_scales)
    occlusion_fractions = _parse_float_list(args.occlusion_fractions)
    recovery_settle_steps = _parse_int_list(args.recovery_settle_steps)
    clean_settle_steps = _parse_int_list(args.clean_settle_steps)

    model, config, checkpoint_hparams = load_generator_checkpoint_model(
        args.checkpoint,
        preset=args.preset,
        seed=args.seed,
    )
    _, _, eval_images, eval_labels = load_mnist_data(
        source=config.data_source,
        dataset_name=config.dataset_name,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        seed=args.seed,
    )
    count = int(args.sample_count)
    labels = (
        jnp.asarray(eval_labels[:count], dtype=jnp.int32)
        if config.conditional
        else None
    )
    probe = compute_generator_state_fitting_probe(
        model,
        eval_images[:count],
        key=jax.random.PRNGKey(args.probe_seed),
        labels=labels,
        image_shape=tuple(int(size) for size in config.image_shape),
        sample_count=count,
        fit_steps=args.fit_steps,
        learning_rate=args.learning_rate,
        init_scale=args.init_scale,
        settle_steps=clean_settle_steps,
        recovery_noise_scales=noise_scales,
        recovery_settle_steps=recovery_settle_steps,
        occlusion_fractions=occlusion_fractions,
        return_images=True,
    )
    images = probe.pop("images", {})
    rows = _condition_rows(
        probe,
        noise_scales=noise_scales,
        recovery_settle_steps=recovery_settle_steps,
        clean_settle_steps=clean_settle_steps,
        occlusion_fractions=occlusion_fractions,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}.csv"
    json_path = output_dir / f"{args.output_prefix}.json"
    contact_path = output_dir / f"{args.output_prefix}_contact_sheet.png"

    write_recovery_rows_csv(rows, csv_path)
    write_json(
        json_path,
        {
            "checkpoint": str(args.checkpoint),
            "preset": args.preset,
            "seed": args.seed,
            "probe_seed": args.probe_seed,
            "noise_scales": noise_scales,
            "occlusion_fractions": occlusion_fractions,
            "recovery_settle_steps": recovery_settle_steps,
            "clean_settle_steps": clean_settle_steps,
            "config": {
                "dataset_name": config.dataset_name,
                "image_shape": config.image_shape,
                "model_family": config.model_family,
                "num_classes": config.num_classes,
                "steps": config.steps,
            },
            "checkpoint_hyperparams": checkpoint_hparams,
            "metrics": probe,
        },
    )
    if images:
        sheet_variants = {
            name: np.asarray(array)
            for name, array in images.items()
        }
        save_state_prior_contact_sheet(
            sheet_variants,
            contact_path,
            image_shape=tuple(int(size) for size in config.image_shape),
        )
        print(f"wrote {contact_path}")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
