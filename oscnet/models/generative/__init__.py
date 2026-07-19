"""Implicit image generators driven by oscillator dynamics."""

from .coarse_horn import (
    CoarseToFineHORNImageGenerator,
    CoarseToFineMultiModeHORNImageGenerator,
)
from .horn import HORNImageGenerator
from .kuramoto import KuramotoImageGenerator
from .multiscale_horn import MultiscaleHORNImageGenerator
from .hybrid import HybridImageGenerator
from .multimode_horn import MultiModeHORNImageGenerator
from .phase_vae import KuramotoPhaseVAE
from .resonator_encoder import (
    GaborStateEncoder,
    RowScanResonatorEncoder,
    build_image_rfb_encoder,
)
from .state_mlp import StateMLPImageGenerator

__all__ = [
    "CoarseToFineHORNImageGenerator",
    "CoarseToFineMultiModeHORNImageGenerator",
    "HORNImageGenerator",
    "KuramotoImageGenerator",
    "KuramotoPhaseVAE",
    "MultiscaleHORNImageGenerator",
    "HybridImageGenerator",
    "MultiModeHORNImageGenerator",
    "StateMLPImageGenerator",
    "GaborStateEncoder",
    "RowScanResonatorEncoder",
    "build_image_rfb_encoder",
]
