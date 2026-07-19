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
            "multimode_horn",
            "multimode_horn_decoder_only",
            "frozen_multimode_horn",
            "coarse_multimode_horn",
            "state_mlp",
            "state_mlp_decoder_only",
            "frozen_state_mlp",
            "hybrid",
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
        choices=["dense", "distance_decay", "local_radius", "fractal"],
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
    parser.add_argument(
        "--state-residual-readout-strength",
        type=float,
        default=0.0,
        help=(
            "Optional final HORN-state local residual readout added to resize-conv "
            "logits. Defaults to 0 to preserve the baseline decoder."
        ),
    )
    parser.add_argument(
        "--state-residual-readout-init-scale",
        type=float,
        default=0.01,
    )
    parser.add_argument("--state-residual-readout-patch-size", type=int, default=5)
    parser.add_argument("--state-residual-readout-sigma", type=float, default=0.0)
    parser.add_argument(
        "--resonant-readout-strength",
        type=float,
        default=0.0,
        help=(
            "Optional shared HORN resonant filter-bank readout from local "
            "phase/velocity/coherence observables. Defaults to 0."
        ),
    )
    parser.add_argument("--resonant-readout-init-scale", type=float, default=0.02)
    parser.add_argument("--resonant-readout-patch-size", type=int, default=5)
    parser.add_argument("--resonant-readout-sigma", type=float, default=0.0)
    parser.add_argument(
        "--state-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only image-to-HORN-state anchor loss weight. 0 disables "
            "the local encoder anchor path."
        ),
    )
    parser.add_argument(
        "--state-anchor-steps",
        type=_parse_int_tuple,
        default=(4, 8, 16),
        help=(
            "Comma-separated HORN settling depths sampled by the state anchor "
            "loss, for example '4,8,16'."
        ),
    )
    parser.add_argument("--state-anchor-noise-scale", type=float, default=0.05)
    parser.add_argument(
        "--state-anchor-mode",
        choices=["none", "reconstruct", "settle", "frozen_dynamics"],
        default="none",
        help=(
            "State-anchor control mode. 'reconstruct' is k=0 encode/decode; "
            "'settle' trains through dynamics; 'frozen_dynamics' stops "
            "gradients on recurrent/conditioning parameters in the anchor path."
        ),
    )
    parser.add_argument("--state-anchor-encoder-kernel-size", type=int, default=3)
    parser.add_argument(
        "--state-anchor-occlusion-fraction",
        type=float,
        default=0.0,
        help=(
            "Total image-area fraction zeroed before anchor encoding as "
            "square occlusion patches. 0 disables occlusion corruption."
        ),
    )
    parser.add_argument(
        "--state-anchor-occlusion-patches",
        type=int,
        default=4,
        help=(
            "Number of square patches the anchor occlusion area is split "
            "into. 1 is a single contiguous block; larger values scatter "
            "the same total area."
        ),
    )
    parser.add_argument(
        "--state-anchor-occlusion-probability",
        type=float,
        default=0.5,
        help="Per-sample probability of applying anchor occlusion corruption.",
    )
    parser.add_argument(
        "--state-anchor-occlusion-curriculum",
        type=_parse_float_tuple,
        default=(),
        help=(
            "Comma-separated occlusion fractions for the corruption "
            "curriculum: each anchor batch samples one fraction uniformly "
            "from this list (augmentation-hardened training). When set it "
            "overrides --state-anchor-occlusion-fraction."
        ),
    )
    parser.add_argument(
        "--state-anchor-clean-weight",
        type=float,
        default=0.0,
        help=(
            "Relative weight of the clean fixed-point anchor term: encode "
            "without corruption, settle, decode, paired MSE. Trains clean "
            "states to survive settling."
        ),
    )
    parser.add_argument(
        "--recovery-eval-sample-count",
        type=int,
        default=0,
        help=(
            "Eval images for the encode-corrupt-settle-decode recovery "
            "metrics (PSNR/SSIM and occluded-region MSE). 0 disables."
        ),
    )
    parser.add_argument(
        "--recovery-eval-noise-scales",
        type=_parse_float_tuple,
        default=(0.25, 0.5),
        help="Comma-separated state-noise scales for recovery eval.",
    )
    parser.add_argument(
        "--recovery-eval-occlusion-fractions",
        type=_parse_float_tuple,
        default=(0.25,),
        help="Comma-separated image-area occlusion fractions for recovery eval.",
    )
    parser.add_argument(
        "--recovery-eval-occlusion-patches",
        type=_parse_int_tuple,
        default=(1, 4),
        help=(
            "Comma-separated patch counts for recovery-eval occlusion, e.g. "
            "'1,4' scores one contiguous block and four scattered patches."
        ),
    )
    parser.add_argument(
        "--recovery-eval-settle-steps",
        type=_parse_int_tuple,
        default=(0, 4, 8, 16),
        help="Comma-separated settle depths for recovery eval.",
    )
    parser.add_argument(
        "--robustness-eval-sample-count",
        type=int,
        default=0,
        help=(
            "Eval images for the graceful-degradation robustness probe "
            "(recovery under weight noise, quantization, and stronger-than-"
            "trained occlusion). 0 disables."
        ),
    )
    parser.add_argument(
        "--robustness-eval-settle-step",
        type=int,
        default=8,
        help="Fixed settle depth used for all robustness-probe conditions.",
    )
    parser.add_argument(
        "--robustness-eval-weight-noise-scales",
        type=_parse_float_tuple,
        default=(0.02, 0.05, 0.1, 0.2),
        help=(
            "Comma-separated per-leaf-std-relative Gaussian weight-noise "
            "scales for the robustness probe."
        ),
    )
    parser.add_argument(
        "--robustness-eval-quant-bits",
        type=_parse_int_tuple,
        default=(8, 6, 4, 3),
        help="Comma-separated weight-quantization bit widths for the probe.",
    )
    parser.add_argument(
        "--robustness-eval-occlusion-fractions",
        type=_parse_float_tuple,
        default=(0.1, 0.25, 0.4, 0.6),
        help=(
            "Comma-separated out-of-distribution occlusion fractions for the "
            "robustness probe."
        ),
    )
    parser.add_argument(
        "--robustness-eval-weight-noise-draws",
        type=int,
        default=3,
        help="Random weight-noise draws averaged per scale in the probe.",
    )
    parser.add_argument(
        "--robustness-eval-heldout-corruptions",
        type=_parse_str_tuple,
        default=(),
        help=(
            "Comma-separated held-out corruption conditions as family:level "
            "(families: gaussian, salt_pepper, stripes), e.g. "
            "'gaussian:0.1,stripes:0.5'. These families never appear in any "
            "training curriculum, so they score generalization to "
            "unanticipated damage."
        ),
    )
    parser.add_argument(
        "--hybrid-router-hidden-dim",
        type=int,
        default=16,
        help="Hidden width of the hybrid model's per-site router MLP.",
    )
    parser.add_argument(
        "--hybrid-router-bias-init",
        type=float,
        default=-1.0,
        help=(
            "Initial router gate bias (logit). Negative starts the hybrid "
            "trusting the free-form path, matching the on-nominal ordering."
        ),
    )
    parser.add_argument(
        "--hybrid-router-mode",
        choices=(
            "learned",
            "fixed_statistic",
            "oracle",
            "free_form",
            "oscillator",
        ),
        default="learned",
        help=(
            "Hybrid routing policy. 'fixed_statistic' is the non-learned "
            "shift/typicality gate (Hybrid Frontier follow-up); 'oracle' "
            "uses an externally supplied per-site mask."
        ),
    )
    parser.add_argument(
        "--hybrid-fixed-gate-scale",
        type=float,
        default=4.0,
        help="Scale for fixed_statistic router residual → gate logit.",
    )
    parser.add_argument(
        "--state-prior-sampling-mode",
        choices=["none", "global", "class"],
        default="none",
        help=(
            "Optional training-time initial-state prior for generated drift "
            "samples. 'global' fits one anchor-state PCA prior across all "
            "classes; 'class' fits one prior per class."
        ),
    )
    parser.add_argument("--state-prior-rank", type=int, default=32)
    parser.add_argument("--state-prior-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--state-prior-refresh-epochs",
        type=int,
        default=1,
        help="Refit the host-side state prior every N epochs when enabled.",
    )
    parser.add_argument(
        "--state-prior-start-epoch",
        type=int,
        default=1,
        help="First epoch that may train generated samples from the state prior.",
    )
    parser.add_argument(
        "--multimode-num-modes",
        type=int,
        default=2,
        help=(
            "Number of frequency modes per spatial HORN site for "
            "model-family=multimode_horn."
        ),
    )
    parser.add_argument(
        "--multimode-frequency-scales",
        type=_parse_float_tuple,
        default=(),
        help=(
            "Comma-separated fixed frequency scales for multimode HORN. "
            "Empty uses a gentle low/high default."
        ),
    )
    parser.add_argument(
        "--multimode-mode-coupling-strength",
        type=float,
        default=0.25,
        help="Within-site coupling strength among multimode HORN frequency bands.",
    )
    parser.add_argument(
        "--multimode-mode-coupling-profile",
        choices=["dense", "adjacent"],
        default="dense",
        help="Within-site mode coupling profile for multimode HORN.",
    )
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
        "--coarse-frequency-scale",
        type=float,
        default=1.0,
        help=(
            "Frequency multiplier for the coarse carrier band. Values < 1 "
            "make the carrier explicitly slower than the fine field."
        ),
    )
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
        "--multiscale-readout-gate-mode",
        choices=["none", "seed_film"],
        default="none",
        help=(
            "Learned coarse-to-fine readout modulation. 'seed_film' uses the "
            "selected auxiliary oscillator layer to scale and shift the "
            "resize-conv seed tensor before rendering."
        ),
    )
    parser.add_argument(
        "--multiscale-readout-gate-strength",
        type=float,
        default=0.0,
        help=(
            "Strength for the learned readout gate. 0 disables the gate even "
            "when --multiscale-readout-gate-mode is set."
        ),
    )
    parser.add_argument(
        "--multiscale-readout-gate-init-scale",
        type=float,
        default=0.0,
        help=(
            "Initialization scale for the readout gate projection. The "
            "default 0 starts from the ungated model while remaining trainable."
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
    parser.add_argument(
        "--frequency-objective-weight",
        type=float,
        default=0.0,
        help=(
            "Optional image-spectrum objective weight. This matches generated "
            "frequency-band and edge statistics to real images without "
            "forcing paired pixel alignment."
        ),
    )
    parser.add_argument(
        "--frequency-objective-edge-weight",
        type=float,
        default=1.0,
        help=(
            "Relative weight for the Laplacian/edge-statistics term inside "
            "--frequency-objective-weight."
        ),
    )
    parser.add_argument(
        "--patch-objective-weight",
        type=float,
        default=0.0,
        help=(
            "Optional local patch-distribution objective weight. This compares "
            "unpaired small image patches with sliced Wasserstein projections."
        ),
    )
    parser.add_argument("--patch-objective-patch-size", type=int, default=5)
    parser.add_argument(
        "--patch-objective-patch-sizes",
        type=_parse_int_tuple,
        default=(),
        help=(
            "Optional comma-separated multiscale patch sizes. Empty uses "
            "--patch-objective-patch-size."
        ),
    )
    parser.add_argument("--patch-objective-stride", type=int, default=4)
    parser.add_argument(
        "--patch-objective-offsets",
        type=_parse_int_tuple,
        default=(0,),
        help=(
            "Comma-separated patch-grid offsets. Multiple offsets score all "
            "offset_y/offset_x combinations and reduce fixed-grid striping."
        ),
    )
    parser.add_argument("--patch-objective-projections", type=int, default=32)
    parser.add_argument(
        "--patch-objective-edge-weight",
        type=float,
        default=0.25,
        help=(
            "Relative weight for the Laplacian-patch term inside "
            "--patch-objective-weight."
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
        "--resize-conv-seed-layout",
        choices=["flat", "retinotopic"],
        default="flat",
        help=(
            "How final oscillator features are arranged before resize-conv. "
            "'flat' preserves the historical reshape; 'retinotopic' maps "
            "oscillator-grid sites to seed pixels."
        ),
    )
    parser.add_argument(
        "--resize-conv-seed-min-channels",
        type=int,
        default=0,
        help=(
            "Minimum retinotopic seed channels. Values above the natural "
            "position/velocity channel count pad local oscillator observables "
            "instead of flattening extra spatial sites."
        ),
    )
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
        "--state-probe-sample-count",
        type=int,
        default=0,
        help=(
            "Samples used for the final linear state-information probe. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--state-probe-target-size",
        type=int,
        default=8,
        help="Low-resolution scaffold size used by the state-information probe.",
    )
    parser.add_argument(
        "--state-probe-ridge",
        type=float,
        default=1e-3,
        help="Ridge penalty for linear probes fitted to traced oscillator states.",
    )
    parser.add_argument(
        "--state-fit-sample-count",
        type=int,
        default=0,
        help=(
            "Number of real images for frozen-decoder per-image state fitting. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--state-fit-steps",
        type=int,
        default=100,
        help="Adam steps for the frozen-decoder state fitting probe.",
    )
    parser.add_argument(
        "--state-fit-learning-rate",
        type=float,
        default=5e-2,
        help="Learning rate for the frozen-decoder state fitting probe.",
    )
    parser.add_argument(
        "--state-fit-init-scale",
        type=float,
        default=0.05,
        help="Initial random state scale for frozen-decoder state fitting.",
    )
    parser.add_argument(
        "--state-fit-ridge",
        type=float,
        default=1e-3,
        help="Ridge penalty for fitted-state linear readout probes.",
    )
    parser.add_argument(
        "--state-fit-settle-steps",
        type=_parse_int_tuple,
        default=(0, 8, 16, 32),
        help=(
            "Comma-separated settling depths to apply after fitting final "
            "states, e.g. '0,8,16,32'."
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
        state_residual_readout_strength=args.state_residual_readout_strength,
        state_residual_readout_init_scale=args.state_residual_readout_init_scale,
        state_residual_readout_patch_size=args.state_residual_readout_patch_size,
        state_residual_readout_sigma=args.state_residual_readout_sigma,
        resonant_readout_strength=args.resonant_readout_strength,
        resonant_readout_init_scale=args.resonant_readout_init_scale,
        resonant_readout_patch_size=args.resonant_readout_patch_size,
        resonant_readout_sigma=args.resonant_readout_sigma,
        state_anchor_weight=args.state_anchor_weight,
        state_anchor_steps=args.state_anchor_steps,
        state_anchor_noise_scale=args.state_anchor_noise_scale,
        state_anchor_mode=args.state_anchor_mode,
        state_anchor_encoder_kernel_size=args.state_anchor_encoder_kernel_size,
        state_anchor_occlusion_fraction=args.state_anchor_occlusion_fraction,
        state_anchor_occlusion_patches=args.state_anchor_occlusion_patches,
        state_anchor_occlusion_probability=args.state_anchor_occlusion_probability,
        state_anchor_occlusion_curriculum=args.state_anchor_occlusion_curriculum,
        state_anchor_clean_weight=args.state_anchor_clean_weight,
        recovery_eval_sample_count=args.recovery_eval_sample_count,
        recovery_eval_noise_scales=args.recovery_eval_noise_scales,
        recovery_eval_occlusion_fractions=args.recovery_eval_occlusion_fractions,
        recovery_eval_occlusion_patches=args.recovery_eval_occlusion_patches,
        recovery_eval_settle_steps=args.recovery_eval_settle_steps,
        robustness_eval_sample_count=args.robustness_eval_sample_count,
        robustness_eval_settle_step=args.robustness_eval_settle_step,
        robustness_eval_weight_noise_scales=(
            args.robustness_eval_weight_noise_scales
        ),
        robustness_eval_quant_bits=args.robustness_eval_quant_bits,
        robustness_eval_occlusion_fractions=(
            args.robustness_eval_occlusion_fractions
        ),
        robustness_eval_weight_noise_draws=args.robustness_eval_weight_noise_draws,
        robustness_eval_heldout_corruptions=(
            args.robustness_eval_heldout_corruptions
        ),
        hybrid_router_hidden_dim=args.hybrid_router_hidden_dim,
        hybrid_router_bias_init=args.hybrid_router_bias_init,
        hybrid_router_mode=args.hybrid_router_mode,
        hybrid_fixed_gate_scale=args.hybrid_fixed_gate_scale,
        state_prior_sampling_mode=args.state_prior_sampling_mode,
        state_prior_rank=args.state_prior_rank,
        state_prior_noise_scale=args.state_prior_noise_scale,
        state_prior_refresh_epochs=args.state_prior_refresh_epochs,
        state_prior_start_epoch=args.state_prior_start_epoch,
        multimode_num_modes=args.multimode_num_modes,
        multimode_frequency_scales=args.multimode_frequency_scales,
        multimode_mode_coupling_strength=args.multimode_mode_coupling_strength,
        multimode_mode_coupling_profile=args.multimode_mode_coupling_profile,
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
        coarse_frequency_scale=args.coarse_frequency_scale,
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
        multiscale_readout_gate_mode=args.multiscale_readout_gate_mode,
        multiscale_readout_gate_strength=args.multiscale_readout_gate_strength,
        multiscale_readout_gate_init_scale=(
            args.multiscale_readout_gate_init_scale
        ),
        coarse_auxiliary_weight=args.coarse_auxiliary_weight,
        coarse_auxiliary_target_size=args.coarse_auxiliary_target_size,
        coarse_auxiliary_loss_mode=args.coarse_auxiliary_loss_mode,
        coarse_readout_consistency_weight=args.coarse_readout_consistency_weight,
        coarse_readout_consistency_onset_epoch=(
            args.coarse_readout_consistency_onset_epoch
        ),
        frequency_objective_weight=args.frequency_objective_weight,
        frequency_objective_edge_weight=args.frequency_objective_edge_weight,
        patch_objective_weight=args.patch_objective_weight,
        patch_objective_patch_size=args.patch_objective_patch_size,
        patch_objective_patch_sizes=args.patch_objective_patch_sizes,
        patch_objective_stride=args.patch_objective_stride,
        patch_objective_offsets=args.patch_objective_offsets,
        patch_objective_projections=args.patch_objective_projections,
        patch_objective_edge_weight=args.patch_objective_edge_weight,
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
        resize_conv_seed_layout=args.resize_conv_seed_layout,
        resize_conv_seed_min_channels=args.resize_conv_seed_min_channels,
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
        state_probe_sample_count=args.state_probe_sample_count,
        state_probe_target_size=args.state_probe_target_size,
        state_probe_ridge=args.state_probe_ridge,
        state_fit_sample_count=args.state_fit_sample_count,
        state_fit_steps=args.state_fit_steps,
        state_fit_learning_rate=args.state_fit_learning_rate,
        state_fit_init_scale=args.state_fit_init_scale,
        state_fit_ridge=args.state_fit_ridge,
        state_fit_settle_steps=args.state_fit_settle_steps,
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
