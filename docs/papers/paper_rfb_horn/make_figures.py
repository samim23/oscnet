#!/usr/bin/env python3
"""Regenerate paper figures from Modal CSV summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
ANALYSIS = ROOT / "outputs" / "analysis"

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(name: str):
    return list(csv.DictReader(open(ANALYSIS / name)))


def mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else float("nan")


def std(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def pick(g, frontend, head, couple, naug):
    for k, v in g.items():
        fe, h, c, n = k
        if fe == frontend and h == head and n == naug and (couple == "" or c == couple):
            return v
    raise KeyError((frontend, head, couple, naug))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load("modal_audio_digit_rfb_plus_tonotopic.csv")
    g = defaultdict(list)
    for r in rows:
        key = (
            r["frontend"],
            r["head_kind"],
            r.get("horn_coupling_kind") or "",
            r.get("train_noise_aug_prob") or "0.0",
        )
        g[key].append(r)

    # Fig 0
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.0, 1.8, 1.2, "Waveform", "#f4f6f7"),
        (2.4, 1.0, 2.2, 1.2, "Learnable RFB\n(ears)", "#d4e6f1"),
        (4.9, 1.0, 2.2, 1.2, "Framed\namplitude", "#d5f5e3"),
        (7.4, 1.0, 2.2, 1.2, "Dense HORN\n(brain)", "#fdebd0"),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y), w, h, facecolor=c, edgecolor="#2c3e50", lw=1.2, zorder=2
            )
        )
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10, zorder=3)
    for x in (2.1, 4.6, 7.1):
        ax.annotate(
            "",
            xy=(x + 0.25, 1.6),
            xytext=(x - 0.05, 1.6),
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.4),
        )
    ax.text(
        5,
        0.35,
        "Rejected here: phase readout · bank AGC · frozen fractal as sufficient · high→low streamlining",
        ha="center",
        fontsize=8,
        color="#7f8c8d",
    )
    ax.set_title("Oscillators as ears and brain", fontsize=12, pad=8)
    fig.savefig(OUT / "fig0_architecture.pdf")
    fig.savefig(OUT / "fig0_architecture.png")
    plt.close()

    # Fig 1
    arms = [
        (("resonator", "horn", "dense", "0.0"), "RFB→dense HORN", "#1b4f72"),
        (("resonator", "horn", "dense_tonotopic", "0.0"), "RFB→tonotopic HORN", "#2874a6"),
        (("resonator", "horn", "fractal_fixed", "0.0"), "RFB→fractal HORN", "#7f8c8d"),
        (("mel", "mlp", "", "0.0"), "mel→MLP", "#b03a2e"),
        (("resonator", "mlp", "", "0.0"), "RFB→MLP", "#ca6f1e"),
    ]
    labels, means, errs, colors = [], [], [], []
    for key, lab, col in arms:
        rs = pick(g, *key)
        labels.append(lab)
        means.append(mean(rs, "final_eval_accuracy"))
        errs.append(std(rs, "final_eval_accuracy"))
        colors.append(col)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = range(len(labels))
    ax.bar(x, means, yerr=errs, color=colors, capsize=3, width=0.72, edgecolor="white")
    ax.axhline(0.1, color="#999", ls="--", lw=0.8, label="chance")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Clean accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Spoken digits · 10k train · matched small capacity")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig1_main_accuracy.pdf")
    fig.savefig(OUT / "fig1_main_accuracy.png")
    plt.close()

    # Fig 2
    metrics = [
        ("final_eval_accuracy", "Clean"),
        ("eval_accuracy_snr10db", "White 10dB"),
        ("eval_accuracy_pink10db", "Pink 10dB"),
        ("eval_accuracy_band10db", "Band 10dB"),
    ]
    series = [
        (("resonator", "horn", "dense", "0.0"), "Dense HORN (clean train)", "#1b4f72"),
        (("resonator", "horn", "dense", "0.4"), "Dense HORN (soft aug)", "#148f77"),
        (("mel", "mlp", "", "0.0"), "mel→MLP", "#b03a2e"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    n_m = len(metrics)
    width = 0.25
    xpos = np.arange(n_m)
    for i, (key, lab, col) in enumerate(series):
        rs = pick(g, *key)
        vals = [mean(rs, m) for m, _ in metrics]
        ax.bar(xpos + (i - 1) * width, vals, width=width, label=lab, color=col, edgecolor="white")
    ax.set_xticks(xpos)
    ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Robustness: soft multi-condition training")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig2_robustness.pdf")
    fig.savefig(OUT / "fig2_robustness.png")
    plt.close()

    # Fig 3
    s3 = load("modal_audio_digit_rfb_stage3.csv")
    g3 = defaultdict(list)
    for r in s3:
        if (
            r["frontend"] == "resonator"
            and r.get("learnable_frontend") == "True"
            and r["head_kind"] == "horn"
        ):
            g3[r.get("readout", "amplitude")].append(r)
    s4 = load("modal_audio_digit_rfb_stage4.csv")
    g4 = defaultdict(list)
    for r in s4:
        if (
            r["frontend"] == "resonator"
            and r.get("learnable_frontend") == "True"
            and r["head_kind"] == "horn"
        ):
            g4[r.get("nonlinearity", "none")].append(r)
    for r in s4:
        if r.get("nonlinearity") == "agc":
            g4["agc"].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), sharey=True)
    labs = ["amplitude", "both", "phase"]
    vals = [mean(g3[k], "final_eval_accuracy") for k in labs]
    axes[0].bar(labs, vals, color=["#1b4f72", "#5d6d7e", "#b03a2e"], edgecolor="white")
    axes[0].set_title("Readout")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)

    labs_n = ["none", "drive_tanh", "envelope_soft", "agc"]
    vals = [mean(g4.get(k, []), "final_eval_accuracy") for k in labs_n]
    axes[1].bar(
        ["linear", "tanh", "envelope", "AGC"],
        vals,
        color=["#1b4f72", "#2874a6", "#7f8c8d", "#b03a2e"],
        edgecolor="white",
    )
    axes[1].set_title("Bank nonlinearity")
    axes[1].tick_params(axis="x", rotation=15)

    keys_c = ["dense", "dense_tonotopic", "fractal_fixed"]
    labs_c = ["dense", "tonotopic", "fractal"]
    vals = []
    for kc in keys_c:
        rs = [
            r
            for r in rows
            if r.get("horn_coupling_kind") == kc
            and r["head_kind"] == "horn"
            and (r.get("train_noise_aug_prob") or "0.0") == "0.0"
        ]
        vals.append(mean(rs, "final_eval_accuracy"))
    axes[2].bar(labs_c, vals, color=["#1b4f72", "#2874a6", "#7f8c8d"], edgecolor="white")
    axes[2].set_title("HORN coupling")
    axes[2].tick_params(axis="x", rotation=15)
    for ax in axes:
        ax.axhline(0.1, color="#bbb", ls="--", lw=0.7)
    fig.suptitle("Ablations: what fails (and what does not)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_ablations.pdf")
    fig.savefig(OUT / "fig3_ablations.png")
    plt.close()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
