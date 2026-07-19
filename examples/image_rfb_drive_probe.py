#!/usr/bin/env python3
"""Stage 0b: cheap local probe of image RFB drive definitions.

Compares frozen Gabor vs row-scan resonator encoders on a synthetic
texture/orientation classification task (no TensorFlow required). Enough to
pick the Stage 1 drive without a Modal night.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax


def make_texture_dataset(
    *,
    n: int,
    seed: int,
    image_size: int = 32,
):
    """10-way synthetic textures: oriented gratings + noise blobs.

    Class id sets grating orientation / spatial frequency — a clean probe for
    whether a Gabor bank beats anisotropic row-scan.
    """

    rng = np.random.default_rng(seed)
    ys = rng.integers(0, 10, size=n)
    xs = np.zeros((n, image_size, image_size), dtype=np.float32)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    for i, lab in enumerate(ys):
        theta = (lab % 5) * (np.pi / 5.0)
        freq = 0.08 + 0.04 * (lab // 5)
        proj = xx * np.cos(theta) + yy * np.sin(theta)
        grating = np.sin(2.0 * np.pi * freq * proj)
        noise = rng.normal(0.0, 0.25, size=(image_size, image_size))
        img = 0.7 * grating + 0.3 * noise
        # random crop jitter via roll
        img = np.roll(img, int(rng.integers(-3, 4)), axis=0)
        img = np.roll(img, int(rng.integers(-3, 4)), axis=1)
        xs[i] = img.astype(np.float32)
    # grayscale → 3ch flat like CIFAR RGB layout for encoder API
    xs = np.stack([xs, xs, xs], axis=1)  # N,C,H,W
    xs = xs.reshape(n, -1)
    return jnp.asarray(xs), jnp.asarray(ys, dtype=jnp.int32)


class ProbeHead(eqx.Module):
    encoder: eqx.Module
    layer: eqx.nn.Linear

    def __init__(self, encoder, *, num_oscillators: int, num_classes: int, key):
        self.encoder = encoder
        self.layer = eqx.nn.Linear(2 * num_oscillators, num_classes, key=key)

    def __call__(self, images):
        pos, vel = self.encoder(images)
        feats = jnp.concatenate([pos, vel], axis=-1)
        return jax.vmap(self.layer)(feats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive", choices=("gabor", "row_scan"), default="gabor")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=500)
    p.add_argument("--num-oscillators", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/image_rfb_drive")
    )
    args = p.parse_args()

    from oscnet.models import build_image_rfb_encoder

    train_x, train_y = make_texture_dataset(
        n=args.train_samples, seed=args.seed
    )
    eval_x, eval_y = make_texture_dataset(
        n=args.eval_samples, seed=args.seed + 99
    )
    drives = ("gabor", "row_scan") if args.sweep else (args.drive,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for drive in drives:
        key = jax.random.PRNGKey(args.seed + (11 if drive == "gabor" else 29))
        k_enc, k_head = jax.random.split(key)
        enc = build_image_rfb_encoder(
            drive=drive,
            num_oscillators=args.num_oscillators,
            image_shape=(32, 32, 3),
            key=k_enc,
        )
        model = ProbeHead(
            enc,
            num_oscillators=args.num_oscillators,
            num_classes=10,
            key=k_head,
        )
        opt = optax.adam(1e-2)
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(model, opt_state, xb, yb):
            def loss_fn(m):
                logits = m(xb)
                return -jnp.mean(
                    jax.nn.log_softmax(logits)[jnp.arange(yb.shape[0]), yb]
                )

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, opt_state = opt.update(
                grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
            )
            return eqx.apply_updates(model, updates), opt_state, loss

        n = int(train_x.shape[0])
        batch = 64
        hist = []
        loss = 0.0
        for epoch in range(args.epochs):
            perm = np.random.default_rng(args.seed + epoch).permutation(n)
            for s in range(0, n, batch):
                idx = perm[s : s + batch]
                if idx.shape[0] < 2:
                    continue
                model, opt_state, loss = step(
                    model, opt_state, train_x[idx], train_y[idx]
                )
            logits = model(eval_x)
            acc = float(jnp.mean(jnp.argmax(logits, axis=-1) == eval_y))
            hist.append(
                {"epoch": epoch, "eval_accuracy": acc, "loss": float(loss)}
            )
        row = {
            "drive": drive,
            "final_eval_accuracy": hist[-1]["eval_accuracy"],
            "chance": 0.1,
            "history": hist,
        }
        rows.append(row)
        print(f"{drive:10s} acc={row['final_eval_accuracy']:.4f}", flush=True)

    out = args.output_dir / "stage0b_summary.json"
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
