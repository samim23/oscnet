"""Load trained MNIST/CIFAR generator checkpoints for offline analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import equinox as eqx
import jax

from oscnet.experiments.mnist_generator import (
    build_mnist_generator_model,
    config_from_args,
    parse_args as parse_generator_args,
)


def load_generator_checkpoint_model(
    checkpoint_path: Path,
    *,
    preset: str,
    seed: int,
) -> Tuple[eqx.Module, Any, Dict[str, Any]]:
    """Build a preset model skeleton and deserialize checkpoint leaves into it.

    The checkpoint file layout is one JSON hyperparameter line followed by
    Equinox-serialized leaves (the format written by the generator runner).
    A few identity-critical hyperparameters are cross-checked against the
    requested preset so the wrong checkpoint fails loudly instead of silently
    producing garbage analyses.
    """

    config = config_from_args(
        parse_generator_args(["--preset", preset, "--seed", str(seed)])
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(seed))
    with Path(checkpoint_path).open("rb") as handle:
        checkpoint_hparams = json.loads(handle.readline().decode())
        model = eqx.tree_deserialise_leaves(handle, model)

    expected = {
        "model_family": config.model_family,
        "dataset_name": config.dataset_name,
        "image_shape": list(config.image_shape),
        "decoder_mode": config.decoder_mode,
    }
    observed = {
        "model_family": checkpoint_hparams.get("model_family"),
        "dataset_name": checkpoint_hparams.get("dataset_name"),
        "image_shape": checkpoint_hparams.get("image_shape"),
        "decoder_mode": checkpoint_hparams.get("decoder_mode"),
    }
    mismatches = {
        key: (expected[key], observed[key])
        for key in expected
        if observed[key] is not None and observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(
            "checkpoint metadata does not match requested preset: "
            f"{mismatches}"
        )
    return model, config, checkpoint_hparams
