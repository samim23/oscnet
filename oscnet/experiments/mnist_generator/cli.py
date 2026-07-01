"""Command-line interface for MNIST oscillator generator experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

from oscnet.experiments.harness import (
    AutoencoderExperimentConfig,
    AutoencoderExperimentResult,
)
from oscnet.experiments.mnist_autoencoder import image_shape_for_dataset

from .config import MNISTGeneratorExperimentConfig
from .presets import GENERATOR_PRESETS, RECOMMENDED_GENERATOR_PRESET, preset_defaults
from .runner import run_mnist_generator_experiment

PRESET_CHOICES = ("none", *tuple(sorted(GENERATOR_PRESETS)))


def _parse_float_tuple(value: str | Sequence[float]) -> Tuple[float, ...]:
    if isinstance(value, str):
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(float(part) for part in value)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_int_tuple(value: str | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(int(part) for part in value)
    if any(step < 0 for step in values):
        raise argparse.ArgumentTypeError("expected non-negative integers")
    return values


def _parse_str_tuple(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, str):
        if value.strip().lower() in ("", "none"):
            return ()
        values = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        values = tuple(str(part).strip() for part in value if str(part).strip())
    return values


def _parse_image_shape(value: str | Sequence[int] | None) -> Tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        parts = tuple(int(part) for part in value)
    if len(parts) not in (2, 3) or any(part < 1 for part in parts):
        raise argparse.ArgumentTypeError(
            "expected image shape like '28,28' or '32,32,3'"
        )
    return parts


def _selected_preset(
    argv: Optional[list[str]],
    *,
    default: str = "none",
) -> str:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default=default,
    )
    args, _ = preset_parser.parse_known_args(argv)
    return args.preset


def build_arg_parser(preset: str = "none") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MNIST image generation with oscillator dynamics."
    )
    parser.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default=preset,
        help=(
            "Named local recipe. The examples default to "
            f"'{RECOMMENDED_GENERATOR_PRESET}', a sparse local HORN setup with "
            "dynamic class coupling."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reference/mnist_generator"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--artifact-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--model-family",
        choices=[
            "kuramoto",
            "decoder_only",
            "frozen_kuramoto",
            "horn",
            "horn_decoder_only",
            "frozen_horn",
            "coarse_horn",
            "coarse_horn_decoder_only",
            "frozen_coarse_horn",
            "multiscale_horn",
            "multiscale_horn_decoder_only",
            "frozen_multiscale_horn",
            "state_mlp",
            "state_mlp_decoder_only",
            "frozen_state_mlp",
        ],
        default="kuramoto",
    )
    parser.add_argument("--num-oscillators", type=int, default=64)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--coupling-strength", type=float, default=1.0)
    parser.add_argument(
        "--main-coupling-strength",
        type=float,
        default=None,
        help=(
            "Optional recurrent oscillator coupling multiplier. Defaults to "
            "--coupling-strength for backward-compatible dynamics; set this "
            "separately to keep class drive fixed while sweeping main coupling."
        ),
    )
    parser.add_argument("--omega-scale", type=float, default=0.2)
    parser.add_argument("--coupling-init-scale", type=float, default=0.05)
    parser.add_argument(
        "--coupling-profile",
        choices=["dense", "distance_decay", "local_radius"],
        default="dense",
    )
    parser.add_argument(
        "--coupling-normalization",
        choices=["none", "row_sum"],
        default="none",
        help=(
            "Optional profile normalization. 'row_sum' keeps every non-empty "
            "row at the same recurrent gain scale, useful when comparing "
            "spatial coupling topologies."
        ),
    )
    parser.add_argument("--coupling-length-scale", type=float, default=0.0)
    parser.add_argument("--coupling-floor", type=float, default=0.0)
    parser.add_argument("--coupling-bias-strength", type=float, default=0.0)
    parser.add_argument("--conditioning-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-target-fraction", type=float, default=1.0)
    parser.add_argument(
        "--conditioning-target-pattern",
        choices=["prefix", "spatial_grid", "center_block"],
        default="prefix",
    )
    parser.add_argument("--horn-frequency", type=float, default=1.0)
    parser.add_argument("--horn-damping", type=float, default=0.15)
    parser.add_argument("--horn-nonlinearity", type=float, default=0.05)
    parser.add_argument("--horn-state-bound", type=float, default=3.0)
    parser.add_argument(
        "--output-feedback-mode",
        choices=["state_proxy", "image"],
        default="state_proxy",
        help=(
            "Feedback source for HORN self-feedback. 'state_proxy' is cheap; "
            "'image' decodes during every settling step and is expensive."
        ),
    )
    parser.add_argument(
        "--output-feedback-strength",
        type=float,
        default=0.0,
        help=(
            "Optional self-feedback from the current decoded image back into "
            "HORN acceleration. Defaults to 0 to preserve plain settling."
        ),
    )
    parser.add_argument("--output-feedback-init-scale", type=float, default=0.02)
    parser.add_argument("--output-feedback-basis-sigma", type=float, default=0.0)
    parser.add_argument("--num-coarse-oscillators", type=int, default=16)
    parser.add_argument(
        "--coarse-coupling-profile",
        choices=["dense", "distance_decay", "local_radius"],
        default="dense",
    )
    parser.add_argument(
        "--coarse-coupling-normalization",
        choices=["none", "row_sum"],
        default="row_sum",
    )
    parser.add_argument("--coarse-coupling-length-scale", type=float, default=0.0)
    parser.add_argument("--coarse-to-fine-strength", type=float, default=1.0)
    parser.add_argument(
        "--coarse-to-fine-profile",
        choices=["dense", "distance_decay", "local_radius"],
        default="dense",
    )
    parser.add_argument(
        "--coarse-to-fine-normalization",
        choices=["none", "row_sum"],
        default="row_sum",
    )
    parser.add_argument("--coarse-to-fine-length-scale", type=float, default=0.0)
    parser.add_argument("--coarse-to-fine-floor", type=float, default=0.0)
    parser.add_argument("--coarse-conditioning-strength", type=float, default=1.0)
    parser.add_argument(
        "--multiscale-layer-sizes",
        type=_parse_int_tuple,
        default=(16, 64),
        help=(
            "Comma-separated auxiliary HORN layer sizes, coarse to fine, "
            "excluding the decoded fine layer. Example: '16,64'."
        ),
    )
    parser.add_argument(
        "--multiscale-frequency-scales",
        type=_parse_float_tuple,
        default=(),
        help=(
            "Optional comma-separated frequency scales for auxiliary layers. "
            "Empty uses a slow-to-mid default."
        ),
    )
    parser.add_argument(
        "--multiscale-coupling-profile",
        choices=["dense", "distance_decay", "local_radius"],
        default="local_radius",
    )
    parser.add_argument(
        "--multiscale-coupling-normalization",
        choices=["none", "row_sum"],
        default="row_sum",
    )
    parser.add_argument("--multiscale-coupling-length-scale", type=float, default=0.0)
    parser.add_argument("--multiscale-coupling-floor", type=float, default=0.0)
    parser.add_argument("--multiscale-vertical-strength", type=float, default=0.25)
    parser.add_argument("--multiscale-feedback-strength", type=float, default=0.0)
    parser.add_argument(
        "--multiscale-vertical-profile",
        choices=["dense", "distance_decay", "local_radius"],
        default="local_radius",
    )
    parser.add_argument(
        "--multiscale-vertical-normalization",
        choices=["none", "row_sum"],
        default="row_sum",
    )
    parser.add_argument("--multiscale-vertical-length-scale", type=float, default=0.0)
    parser.add_argument("--multiscale-vertical-floor", type=float, default=0.0)
    parser.add_argument("--multiscale-vertical-phase-lag", type=float, default=0.0)
    parser.add_argument("--multiscale-feedback-phase-lag", type=float, default=0.0)
    parser.add_argument(
        "--multiscale-vertical-signal-scale",
        type=float,
        default=1.0,
        help=(
            "Post-profile scale for vertical drive/modulation. Use this to "
            "calibrate whether the vertical route is causal before adding "
            "deeper hierarchy."
        ),
    )
    parser.add_argument(
        "--multiscale-feedback-signal-mode",
        choices=["position", "state"],
        default="position",
        help=(
            "Signal used by bottom-up multiscale feedback projections. "
            "'position' preserves the historical phase/position-only route; "
            "'state' feeds back bounded position-plus-velocity evidence."
        ),
    )
    parser.add_argument(
        "--multiscale-feedback-source-gate",
        choices=["all", "conditioning", "non_conditioning", "weighted"],
        default="all",
        help=(
            "Optional source-side gate for bottom-up feedback projections "
            "from the decoded fine layer. 'conditioning' lets the coarse "
            "layer listen only to class-drive target columns; "
            "'non_conditioning' listens to the complement; 'weighted' uses "
            "--multiscale-feedback-source-mix."
        ),
    )
    parser.add_argument(
        "--multiscale-feedback-source-mix",
        type=_parse_float_tuple,
        default=(1.0, 1.0),
        help=(
            "Two comma-separated weights for weighted bottom-up source "
            "gating: conditioning,non_conditioning. The weighted gate is "
            "mean-normalized so average feedback strength stays comparable."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-target-gate",
        choices=["all", "conditioning", "non_conditioning"],
        default="all",
        help=(
            "Optional target-side gate for vertical projections into the fine "
            "layer. 'conditioning' drives only the class-drive target mask; "
            "'non_conditioning' drives the complement."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-soft-gate-floor",
        type=float,
        default=0.0,
        help=(
            "Floor for selective vertical target gates. A value of 0 keeps "
            "binary routing; 0.25 gives non-target fine columns one quarter "
            "of the "
            "vertical profile."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-mode",
        choices=["additive", "gain_modulation", "signed_gain", "dual_gain"],
        default="additive",
        help=(
            "How vertical projections affect target HORN layers. 'additive' "
            "adds a source-minus-target acceleration term; 'gain_modulation' "
            "uses the source projection as a bounded gain on local recurrent "
            "and class-conditioning dynamics; 'signed_gain' allows the "
            "bounded gain to become inhibitory; 'dual_gain' combines broad "
            "nonnegative gain with selective signed modulation."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-gain-target",
        choices=["drive", "coupling", "conditioning", "damping"],
        default="drive",
        help=(
            "Fine-layer HORN term targeted by non-additive vertical gain. "
            "'drive' matches the original behavior and scales local coupling "
            "plus class conditioning; 'coupling' scales only local recurrent "
            "interaction; 'conditioning' scales only class drive; 'damping' "
            "uses the vertical signal as a nonnegative damping gain."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-gain-normalization",
        choices=["none", "center", "center_rms"],
        default="none",
        help=(
            "Optional homeostatic normalization for non-additive vertical "
            "gain signals. 'center' keeps mean gain near one; 'center_rms' "
            "also rescales centered modulation to "
            "--multiscale-vertical-gain-target-std."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-gain-target-std",
        type=float,
        default=0.0,
        help=(
            "Target per-sample modulation RMS for "
            "--multiscale-vertical-gain-normalization=center_rms. A value "
            "of 0 keeps the centered signal's original RMS."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-broad-gain-scale",
        type=float,
        default=1.0,
        help="Broad-route scale used only by multiscale-vertical-mode=dual_gain.",
    )
    parser.add_argument(
        "--multiscale-vertical-selective-gain-scale",
        type=float,
        default=1.0,
        help=(
            "Selective signed-route scale used only by "
            "multiscale-vertical-mode=dual_gain."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-schedule",
        choices=["constant", "delayed", "linear_ramp"],
        default="constant",
        help=(
            "Step schedule for vertical drive/gain during HORN settling. "
            "'constant' matches the original behavior; 'delayed' turns the "
            "vertical route on at --multiscale-vertical-onset-step; "
            "'linear_ramp' ramps from zero after onset over "
            "--multiscale-vertical-ramp-steps."
        ),
    )
    parser.add_argument(
        "--multiscale-vertical-onset-step",
        type=int,
        default=0,
        help="First settling step where delayed/ramped vertical gain can act.",
    )
    parser.add_argument(
        "--multiscale-vertical-ramp-steps",
        type=int,
        default=0,
        help=(
            "Number of settling steps for linear_ramp vertical gain. Values "
            "below 1 are treated as an immediate one-step ramp."
        ),
    )
    parser.add_argument("--multiscale-conditioning-strength", type=float, default=1.0)
    parser.add_argument(
        "--multiscale-auxiliary-readout-layer",
        type=int,
        default=0,
        help=(
            "Auxiliary multiscale layer decoded for the optional coarse "
            "low-resolution image loss. Negative indices count from the last "
            "auxiliary layer."
        ),
    )
    parser.add_argument(
        "--multiscale-readout-fusion-strength",
        type=float,
        default=0.0,
        help=(
            "Blend strength for injecting the upsampled auxiliary readout into "
            "the final image. 0 disables fusion; small values such as 0.1 "
            "test whether the coarse scaffold helps rendering."
        ),
    )
    parser.add_argument(
        "--coarse-auxiliary-weight",
        type=float,
        default=0.0,
        help=(
            "Optional low-resolution auxiliary image loss weight for "
            "multiscale HORN. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--coarse-auxiliary-target-size",
        type=int,
        default=8,
        help="Square target size for the optional coarse auxiliary image loss.",
    )
    parser.add_argument(
        "--coarse-auxiliary-loss-mode",
        choices=["mse", "distributional"],
        default="mse",
        help=(
            "Objective for the optional coarse auxiliary image loss. 'mse' "
            "matches the historical paired low-resolution target; "
            "'distributional' matches low-resolution moments/marginals and "
            "class moments to preserve generative diversity."
        ),
    )
    parser.add_argument(
        "--coarse-readout-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only weight that matches the downsampled final image to "
            "the same-trajectory auxiliary readout. This lets the coarse layer "
            "guide the fine readout without direct output blending."
        ),
    )
    parser.add_argument(
        "--coarse-readout-consistency-onset-epoch",
        type=int,
        default=0,
        help=(
            "Epoch at which coarse readout consistency turns on. Use a small "
            "warmup so the auxiliary scaffold learns before guiding the final "
            "readout. 0 enables it from the start."
        ),
    )
    parser.add_argument("--state-mlp-hidden-dim", type=int, default=48)
    parser.add_argument("--state-mlp-depth", type=int, default=1)
    parser.add_argument("--state-mlp-residual-scale", type=float, default=0.1)
    parser.add_argument(
        "--train-recurrent-dynamics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--train-conditioning-dynamics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--label-phase-scale", type=float, default=0.5)
    parser.add_argument("--num-condition-oscillators", type=int, default=0)
    parser.add_argument(
        "--conditioning-mode",
        choices=["none", "phase_shift", "class_coupling", "class_oscillator"],
        default="phase_shift",
    )
    parser.add_argument(
        "--readout-mode",
        choices=["absolute", "relative", "ref_oscillator", "mean_relative"],
        default="absolute",
    )
    parser.add_argument(
        "--decoder-mode",
        choices=["mlp", "spatial_basis", "local_basis", "resize_conv"],
        default="mlp",
    )
    parser.add_argument(
        "--image-shape",
        type=_parse_image_shape,
        default=None,
        help=(
            "Flat image shape as 'height,width' or 'height,width,channels'. "
            "Defaults to the shape implied by --dataset-name."
        ),
    )
    parser.add_argument("--spatial-basis-sigma", type=float, default=0.0)
    parser.add_argument("--local-patch-size", type=int, default=5)
    parser.add_argument("--resize-conv-seed-size", type=int, default=7)
    parser.add_argument("--resize-conv-upsamples", type=int, default=2)
    parser.add_argument("--resize-conv-min-channels", type=int, default=8)
    parser.add_argument(
        "--output-activation",
        choices=["identity", "sigmoid", "tanh"],
        default="sigmoid",
    )
    parser.add_argument("--num-projections", type=int, default=64)
    parser.add_argument("--moment-weight", type=float, default=0.1)
    parser.add_argument("--pixel-marginal-weight", type=float, default=1.0)
    parser.add_argument("--class-moment-weight", type=float, default=0.0)
    parser.add_argument("--prototype-weight", type=float, default=0.0)
    parser.add_argument(
        "--loss-mode",
        choices=[
            "distributional",
            "pixel_drift",
            "feature_drift",
            "pixel_feature_drift",
        ],
        default="distributional",
    )
    parser.add_argument("--pixel-drift-weight", type=float, default=1.0)
    parser.add_argument("--feature-drift-weight", type=float, default=1.0)
    parser.add_argument(
        "--feature-drift-mode",
        choices=["none", "structural", "learned"],
        default="structural",
    )
    parser.add_argument("--learned-feature-epochs", type=int, default=0)
    parser.add_argument(
        "--learned-feature-kind",
        choices=["mlp", "conv", "residual_conv"],
        default="mlp",
        help=(
            "Classifier architecture for learned feature-drift training. "
            "Use 'residual_conv' for stronger CIFAR semantic feature drift."
        ),
    )
    parser.add_argument("--learned-feature-dim", type=int, default=128)
    parser.add_argument("--learned-feature-depth", type=int, default=2)
    parser.add_argument("--learned-feature-learning-rate", type=float, default=1e-3)
    parser.add_argument("--learned-feature-weight-decay", type=float, default=1e-4)
    parser.add_argument("--quality-classifier-epochs", type=int, default=0)
    parser.add_argument(
        "--quality-classifier-kind",
        choices=["mlp", "conv", "residual_conv"],
        default="mlp",
        help=(
            "Classifier architecture for generated-label quality metrics. "
            "Use 'conv' or 'residual_conv' for image datasets where the flat "
            "MLP judge is weak."
        ),
    )
    parser.add_argument("--quality-classifier-dim", type=int, default=128)
    parser.add_argument("--quality-classifier-depth", type=int, default=2)
    parser.add_argument("--quality-classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--quality-classifier-weight-decay", type=float, default=1e-4)
    parser.add_argument("--quality-classifier-train-limit", type=int, default=None)
    parser.add_argument("--quality-classifier-eval-limit", type=int, default=None)
    parser.add_argument("--drift-queue-size", type=int, default=0)
    parser.add_argument("--drift-queue-num-pos", type=int, default=0)
    parser.add_argument("--distributional-weight", type=float, default=0.0)
    parser.add_argument("--drift-gamma", type=float, default=0.2)
    parser.add_argument(
        "--drift-temperatures",
        type=_parse_float_tuple,
        default=(0.02, 0.05, 0.2),
        help="Comma-separated drift temperatures, e.g. '0.02,0.05,0.2'.",
    )
    parser.add_argument("--output-bias-init", type=float, default=-2.0)
    parser.add_argument("--eval-sample-count", type=int, default=128)
    parser.add_argument(
        "--attractor-variants-per-class",
        type=int,
        default=4,
        help=(
            "Number of same-label initial-state samples used for the final "
            "attractor robustness diagnostic. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--vertical-audit-modes",
        type=_parse_str_tuple,
        default=(),
        help=(
            "Comma-separated sample-time vertical intervention modes to audit "
            "after training, e.g. 'normal,zero,shuffle,flip,scale025'."
        ),
    )
    parser.add_argument(
        "--vertical-audit-sample-count",
        type=int,
        default=0,
        help=(
            "Samples used for vertical intervention quality metrics. "
            "Set to 0 to reuse --eval-sample-count."
        ),
    )
    parser.add_argument(
        "--train-settling-steps",
        type=_parse_int_tuple,
        default=(),
        help=(
            "Comma-separated settling depths to average during training, "
            "for example '4,8,16'. Empty means train only at --steps."
        ),
    )
    parser.add_argument(
        "--settling-steps",
        type=_parse_int_tuple,
        default=(),
        help=(
            "Comma-separated test-time settling depths to score after training, "
            "for example '0,1,2,4,8,16,32'."
        ),
    )
    parser.add_argument(
        "--data-source",
        choices=["tfds", "idx", "synthetic"],
        default="idx",
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset",
        choices=["mnist", "fashion_mnist", "cifar10_gray", "cifar10_rgb"],
        default="mnist",
        help=(
            "Flat image dataset to load. MNIST and Fashion-MNIST use IDX; "
            "CIFAR-10 can be loaded as grayscale or channel-first RGB."
        ),
    )
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--eval-limit", type=int, default=1_000)
    parser.set_defaults(_preset_defaults_applied=False)
    if preset != "none":
        defaults = preset_defaults(preset)
        defaults["preset"] = preset
        defaults["_preset_defaults_applied"] = True
        parser.set_defaults(**defaults)
    return parser


def config_from_args(args: argparse.Namespace) -> MNISTGeneratorExperimentConfig:
    if (
        getattr(args, "preset", "none") != "none"
        and not getattr(args, "_preset_defaults_applied", False)
    ):
        raise ValueError(
            "preset defaults were not applied; use parse_args(argv), main(argv), "
            "or build_arg_parser(preset_name) before config_from_args"
        )
    run = AutoencoderExperimentConfig(
        name="mnist_generator",
        output_dir=Path(args.output_dir),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        artifact_every=args.artifact_every,
    )
    image_shape = (
        args.image_shape
        if args.image_shape is not None
        else image_shape_for_dataset(args.dataset_name)
    )
    return MNISTGeneratorExperimentConfig(
        run=run,
        model_family=args.model_family,
        num_oscillators=args.num_oscillators,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_depth=args.decoder_depth,
        steps=args.steps,
        dt=args.dt,
        coupling_strength=args.coupling_strength,
        main_coupling_strength=args.main_coupling_strength,
        omega_scale=args.omega_scale,
        coupling_init_scale=args.coupling_init_scale,
        coupling_profile=args.coupling_profile,
        coupling_normalization=args.coupling_normalization,
        coupling_length_scale=args.coupling_length_scale,
        coupling_floor=args.coupling_floor,
        coupling_bias_strength=args.coupling_bias_strength,
        conditioning_strength=args.conditioning_strength,
        conditioning_target_fraction=args.conditioning_target_fraction,
        conditioning_target_pattern=args.conditioning_target_pattern,
        horn_frequency=args.horn_frequency,
        horn_damping=args.horn_damping,
        horn_nonlinearity=args.horn_nonlinearity,
        horn_state_bound=args.horn_state_bound,
        output_feedback_mode=args.output_feedback_mode,
        output_feedback_strength=args.output_feedback_strength,
        output_feedback_init_scale=args.output_feedback_init_scale,
        output_feedback_basis_sigma=args.output_feedback_basis_sigma,
        num_coarse_oscillators=args.num_coarse_oscillators,
        coarse_coupling_profile=args.coarse_coupling_profile,
        coarse_coupling_normalization=args.coarse_coupling_normalization,
        coarse_coupling_length_scale=args.coarse_coupling_length_scale,
        coarse_to_fine_strength=args.coarse_to_fine_strength,
        coarse_to_fine_profile=args.coarse_to_fine_profile,
        coarse_to_fine_normalization=args.coarse_to_fine_normalization,
        coarse_to_fine_length_scale=args.coarse_to_fine_length_scale,
        coarse_to_fine_floor=args.coarse_to_fine_floor,
        coarse_conditioning_strength=args.coarse_conditioning_strength,
        multiscale_layer_sizes=args.multiscale_layer_sizes,
        multiscale_frequency_scales=args.multiscale_frequency_scales,
        multiscale_coupling_profile=args.multiscale_coupling_profile,
        multiscale_coupling_normalization=args.multiscale_coupling_normalization,
        multiscale_coupling_length_scale=args.multiscale_coupling_length_scale,
        multiscale_coupling_floor=args.multiscale_coupling_floor,
        multiscale_vertical_strength=args.multiscale_vertical_strength,
        multiscale_feedback_strength=args.multiscale_feedback_strength,
        multiscale_vertical_profile=args.multiscale_vertical_profile,
        multiscale_vertical_normalization=args.multiscale_vertical_normalization,
        multiscale_vertical_length_scale=args.multiscale_vertical_length_scale,
        multiscale_vertical_floor=args.multiscale_vertical_floor,
        multiscale_vertical_phase_lag=args.multiscale_vertical_phase_lag,
        multiscale_feedback_phase_lag=args.multiscale_feedback_phase_lag,
        multiscale_vertical_signal_scale=args.multiscale_vertical_signal_scale,
        multiscale_feedback_signal_mode=args.multiscale_feedback_signal_mode,
        multiscale_feedback_source_gate=args.multiscale_feedback_source_gate,
        multiscale_feedback_source_mix=args.multiscale_feedback_source_mix,
        multiscale_vertical_target_gate=args.multiscale_vertical_target_gate,
        multiscale_vertical_soft_gate_floor=(
            args.multiscale_vertical_soft_gate_floor
        ),
        multiscale_vertical_mode=args.multiscale_vertical_mode,
        multiscale_vertical_gain_target=args.multiscale_vertical_gain_target,
        multiscale_vertical_gain_normalization=(
            args.multiscale_vertical_gain_normalization
        ),
        multiscale_vertical_gain_target_std=(
            args.multiscale_vertical_gain_target_std
        ),
        multiscale_vertical_broad_gain_scale=(
            args.multiscale_vertical_broad_gain_scale
        ),
        multiscale_vertical_selective_gain_scale=(
            args.multiscale_vertical_selective_gain_scale
        ),
        multiscale_vertical_schedule=args.multiscale_vertical_schedule,
        multiscale_vertical_onset_step=args.multiscale_vertical_onset_step,
        multiscale_vertical_ramp_steps=args.multiscale_vertical_ramp_steps,
        multiscale_conditioning_strength=args.multiscale_conditioning_strength,
        multiscale_auxiliary_readout_layer=(
            args.multiscale_auxiliary_readout_layer
        ),
        multiscale_readout_fusion_strength=(
            args.multiscale_readout_fusion_strength
        ),
        coarse_auxiliary_weight=args.coarse_auxiliary_weight,
        coarse_auxiliary_target_size=args.coarse_auxiliary_target_size,
        coarse_auxiliary_loss_mode=args.coarse_auxiliary_loss_mode,
        coarse_readout_consistency_weight=args.coarse_readout_consistency_weight,
        coarse_readout_consistency_onset_epoch=(
            args.coarse_readout_consistency_onset_epoch
        ),
        state_mlp_hidden_dim=args.state_mlp_hidden_dim,
        state_mlp_depth=args.state_mlp_depth,
        state_mlp_residual_scale=args.state_mlp_residual_scale,
        train_recurrent_dynamics=args.train_recurrent_dynamics,
        train_conditioning_dynamics=args.train_conditioning_dynamics,
        conditional=args.conditional,
        num_classes=args.num_classes,
        label_phase_scale=args.label_phase_scale,
        num_condition_oscillators=args.num_condition_oscillators,
        conditioning_mode=args.conditioning_mode,
        readout_mode=args.readout_mode,
        decoder_mode=args.decoder_mode,
        image_shape=image_shape,
        spatial_basis_sigma=args.spatial_basis_sigma,
        local_patch_size=args.local_patch_size,
        resize_conv_seed_size=args.resize_conv_seed_size,
        resize_conv_upsamples=args.resize_conv_upsamples,
        resize_conv_min_channels=args.resize_conv_min_channels,
        output_activation=args.output_activation,
        output_bias_init=args.output_bias_init,
        num_projections=args.num_projections,
        moment_weight=args.moment_weight,
        pixel_marginal_weight=args.pixel_marginal_weight,
        class_moment_weight=args.class_moment_weight,
        prototype_weight=args.prototype_weight,
        loss_mode=args.loss_mode,
        pixel_drift_weight=args.pixel_drift_weight,
        feature_drift_weight=args.feature_drift_weight,
        feature_drift_mode=args.feature_drift_mode,
        learned_feature_epochs=args.learned_feature_epochs,
        learned_feature_kind=args.learned_feature_kind,
        learned_feature_dim=args.learned_feature_dim,
        learned_feature_depth=args.learned_feature_depth,
        learned_feature_learning_rate=args.learned_feature_learning_rate,
        learned_feature_weight_decay=args.learned_feature_weight_decay,
        quality_classifier_epochs=args.quality_classifier_epochs,
        quality_classifier_kind=args.quality_classifier_kind,
        quality_classifier_dim=args.quality_classifier_dim,
        quality_classifier_depth=args.quality_classifier_depth,
        quality_classifier_learning_rate=args.quality_classifier_learning_rate,
        quality_classifier_weight_decay=args.quality_classifier_weight_decay,
        quality_classifier_train_limit=args.quality_classifier_train_limit,
        quality_classifier_eval_limit=args.quality_classifier_eval_limit,
        drift_queue_size=args.drift_queue_size,
        drift_queue_num_pos=args.drift_queue_num_pos,
        distributional_weight=args.distributional_weight,
        drift_gamma=args.drift_gamma,
        drift_temperatures=args.drift_temperatures,
        eval_sample_count=args.eval_sample_count,
        attractor_variants_per_class=args.attractor_variants_per_class,
        vertical_audit_modes=args.vertical_audit_modes,
        vertical_audit_sample_count=args.vertical_audit_sample_count,
        train_settling_steps=args.train_settling_steps,
        settling_steps=args.settling_steps,
        dataset_name=args.dataset_name,
        data_source=args.data_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )


def parse_args(
    argv: Optional[list[str]] = None,
    *,
    default_preset: str = "none",
) -> argparse.Namespace:
    """Parse CLI arguments with preset defaults applied before user overrides."""

    preset = _selected_preset(argv, default=default_preset)
    return build_arg_parser(preset).parse_args(argv)


def main(
    argv: Optional[list[str]] = None,
    *,
    default_preset: str = "none",
) -> AutoencoderExperimentResult:
    return run_mnist_generator_experiment(
        config_from_args(parse_args(argv, default_preset=default_preset))
    )


if __name__ == "__main__":
    main()
