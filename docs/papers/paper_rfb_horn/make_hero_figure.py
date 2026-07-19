#!/usr/bin/env python3
"""Multi-panel hero figure from real OscNet math + run logs."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
ANALYSIS = ROOT / "outputs" / "analysis" / "modal_audio_digit_rfb_plus_controls.json"

# Palette aligned with AudioPrism-ish green / teal / maroon
C_RFB = "#2e8b57"
C_HORN = "#2e86ab"
C_OUT = "#a23b72"
C_MEL = "#b03a2e"
C_MLP = "#ca6f1e"
C_MUTE = "#95a5a6"
C_GRU = "#1a5276"


def dho_impulse(gamma: float, omega: float, t: np.ndarray) -> np.ndarray:
    """Underdamped DHO impulse response (analytic envelope)."""
    om = max(omega, 1e-3)
    g = max(gamma, 1e-6)
    wd = np.sqrt(max(om**2 - g**2, 1e-8))
    return np.exp(-g * t) * np.sin(wd * t)


def bank_gains(n_bands: int = 16, q: float = 4.0, sr: float = 16000.0):
    from oscnet.core.resonators import log_spaced_omegas

    omegas = np.asarray(
        log_spaced_omegas(
            n_bands, f_min_hz=100.0, f_max_hz=7000.0, sample_rate=sr
        )
    )
    gammas = omegas / (2.0 * q)
    # Frequency-domain power response |H(jw)|^2 for driven DHO
    freqs = np.logspace(np.log10(80), np.log10(sr / 2.2), 800)
    w = 2 * np.pi * freqs
    gains = []
    for om, g in zip(omegas, gammas):
        # H(s) = alpha / (s^2 + 2g s + om^2); |H(jw)|
        denom = (om**2 - w**2) ** 2 + (2 * g * w) ** 2
        mag = (om**2) / np.sqrt(denom + 1e-24)  # unit-ish peak scaling
        gains.append(mag / (mag.max() + 1e-12))
    return freqs, np.stack(gains, axis=0), omegas


def load_histories():
    rows = json.load(open(ANALYSIS))["rows"]
    want = {
        ("resonator", "horn"): "RFB→HORN",
        ("resonator", "gru"): "RFB→GRU",
        ("mel", "horn"): "mel→HORN",
        ("mel", "gru"): "mel→GRU",
        ("mel", "mlp"): "mel→MLP",
        ("resonator", "mlp"): "RFB→MLP",
    }
    arms: dict[str, dict] = {}
    for r in rows:
        if int(r.get("seed", -1)) != 0 or not r.get("history"):
            continue
        key = (r.get("frontend"), r.get("head_kind"))
        name = want.get(key)
        if name and name not in arms:
            arms[name] = r
    return arms


def draw_architecture(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("A  Architecture", loc="left", fontsize=11, fontweight="bold")

    # waveform stub
    tw = np.linspace(0, 1, 80)
    wave = 0.25 * np.sin(2 * np.pi * 6 * tw) * np.exp(-1.5 * (tw - 0.4) ** 2)
    ax.plot(0.3 + tw * 1.4, 2.0 + wave, color="black", lw=1.0)
    ax.text(1.0, 2.7, "waveform", ha="center", fontsize=8, color="#333")

    # RFB box
    rfb = mpatches.FancyBboxPatch(
        (2.2, 1.1), 2.6, 1.8, boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor="#d5f5e3", edgecolor=C_RFB, lw=1.5,
    )
    ax.add_patch(rfb)
    ax.text(3.5, 2.55, "RFB", ha="center", fontsize=11, fontweight="bold", color=C_RFB)
    ax.text(3.5, 2.15, "learnable resonators", ha="center", fontsize=7, color="#1e8449")
    for i, y in enumerate(np.linspace(1.35, 1.85, 5)):
        ax.add_patch(plt.Circle((2.7 + i * 0.4, y + 0.15), 0.12, color=C_RFB, alpha=0.35 + 0.1 * i))
    ax.text(3.5, 1.25, r"$\{\omega_k, Q_k\}$", ha="center", fontsize=8, color=C_RFB)

    # HORN cloud-ish box
    horn = mpatches.FancyBboxPatch(
        (5.6, 1.0), 3.0, 2.0, boxstyle="round,pad=0.03,rounding_size=0.2",
        facecolor="#d6eaf8", edgecolor=C_HORN, lw=1.5,
    )
    ax.add_patch(horn)
    ax.text(7.1, 2.65, "HORN", ha="center", fontsize=11, fontweight="bold", color=C_HORN)
    ax.text(7.1, 2.3, "dense recurrent coupling", ha="center", fontsize=7, color="#1a5276")
    rng = np.random.default_rng(0)
    xs = rng.uniform(5.95, 8.2, 12)
    ys = rng.uniform(1.25, 2.05, 12)
    ax.scatter(xs, ys, s=40, c=C_HORN, alpha=0.55, zorder=3)
    for i in range(10):
        ax.annotate(
            "",
            xy=(xs[(i + 3) % 12], ys[(i + 3) % 12]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-", color=C_HORN, alpha=0.25, lw=0.8),
        )

    # readout
    for i, dig in enumerate(["0", "1", "…", "6", "…", "9"]):
        y = 1.2 + i * 0.35
        ax.add_patch(plt.Circle((10.3, y), 0.16, color=C_OUT if dig == "6" else "#f5b7b1", ec=C_OUT, lw=1))
        ax.text(10.3, y, dig, ha="center", va="center", fontsize=7, color="white" if dig == "6" else "#641e16")
    ax.text(10.3, 3.5, "logits", ha="center", fontsize=8, color=C_OUT)

    for x0, x1 in [(1.8, 2.15), (4.9, 5.5), (8.7, 9.9)]:
        ax.annotate(
            "",
            xy=(x1, 2.0),
            xytext=(x0, 2.0),
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.3),
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )

    fig = plt.figure(figsize=(11.0, 7.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0], hspace=0.35, wspace=0.32)

    # A architecture
    ax_a = fig.add_subplot(gs[0, :2])
    draw_architecture(ax_a)

    # B DHO dynamics
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.set_title("B  Resonator dynamics", loc="left", fontsize=11, fontweight="bold")
    t = np.linspace(0, 0.08, 400)
    ax_b.plot(t * 1000, dho_impulse(30, 2 * np.pi * 400, t), color=C_RFB, lw=1.4, label=r"high $\omega$, light $\gamma$")
    ax_b.plot(t * 1000, dho_impulse(80, 2 * np.pi * 200, t), color="#1a5276", lw=1.2, label=r"low $\omega$, heavy $\gamma$")
    ax_b.set_xlabel("time (ms)")
    ax_b.set_ylabel(r"$x(t)$")
    ax_b.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax_b.text(
        0.02,
        0.02,
        r"$\ddot x + 2\gamma\dot x + \omega^2 x = \alpha I(t)$",
        transform=ax_b.transAxes,
        fontsize=8,
        color="#333",
    )

    # C bank gains
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("C  RFB frequency responses", loc="left", fontsize=11, fontweight="bold")
    freqs, gains, omegas = bank_gains()
    for i, g in enumerate(gains):
        ax_c.plot(freqs, g, color=C_RFB, alpha=0.35 + 0.4 * (i / max(len(gains) - 1, 1)), lw=1.0)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("frequency (Hz)")
    ax_c.set_ylabel("normalized gain")
    ax_c.set_xlim(80, 8000)
    ax_c.set_ylim(0, 1.05)

    # D toy signal path
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("D  Tonotopic band envelopes", loc="left", fontsize=11, fontweight="bold")
    sr = 16000
    t = np.arange(0, 0.6, 1 / sr)
    # simple "digit-like" chirp burst
    env = np.exp(-((t - 0.25) ** 2) / (2 * 0.04**2))
    sig = env * np.sin(2 * np.pi * (300 + 1200 * t) * t)
    # band envelopes via FFT power in band windows
    from numpy.fft import rfft, rfftfreq

    n_win = 32
    hop = len(sig) // n_win
    centers = omegas / (2 * np.pi)
    # pick 3 bands: low mid high
    idxs = [2, len(centers) // 2, len(centers) - 3]
    colors = ["#5dade2", C_HORN, "#1a5276"]
    labels = ["low band", "mid band", "high band"]
    ax_d.plot(t, 0.55 + 0.35 * sig / (np.max(np.abs(sig)) + 1e-9), color="black", lw=0.7, label="input")
    for k, (bi, col, lab) in enumerate(zip(idxs, colors, labels)):
        f0 = centers[bi]
        # simple bandpass energy over frames
        frames = []
        ts = []
        for i in range(n_win):
            sl = sig[i * hop : (i + 1) * hop]
            if len(sl) < 8:
                continue
            spec = np.abs(rfft(sl * np.hanning(len(sl))))
            ff = rfftfreq(len(sl), 1 / sr)
            bw = max(f0 / 4, 40)
            mask = (ff > f0 - bw) & (ff < f0 + bw)
            frames.append(float(np.sqrt(np.mean(spec[mask] ** 2) + 1e-12)))
            ts.append((i + 0.5) * hop / sr)
        frames = np.asarray(frames)
        frames = frames / (frames.max() + 1e-9)
        ax_d.plot(ts, 0.05 + 0.28 * frames + 0.12 * k, color=col, lw=1.3, label=lab)
    ax_d.set_xlabel("time (s)")
    ax_d.set_yticks([])
    ax_d.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax_d.set_xlim(0, 0.6)

    # E training curves from fair-control Modal histories
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.set_title("E  Fair controls (seed 0)", loc="left", fontsize=11, fontweight="bold")
    arms = load_histories()
    styles = {
        "mel→GRU": (C_MEL, 1.8),
        "mel→HORN": (C_HORN, 1.8),
        "RFB→GRU": (C_GRU, 1.6),
        "RFB→HORN": (C_RFB, 1.8),
        "mel→MLP": (C_MLP, 1.2),
        "RFB→MLP": (C_MUTE, 1.2),
    }
    for name, (col, lw) in styles.items():
        r = arms.get(name)
        if not r:
            continue
        hist = r["history"]
        xs = [h["epoch"] + 1 for h in hist]
        ys = [100 * float(h["eval_accuracy"]) for h in hist]
        ax_e.plot(xs, ys, color=col, lw=lw, label=name)
    ax_e.axhline(10, color=C_MUTE, ls="--", lw=0.8)
    ax_e.set_xlabel("# training epochs")
    ax_e.set_ylabel("eval accuracy (%)")
    ax_e.set_ylim(0, 100)
    ax_e.legend(fontsize=6.0, frameon=False, loc="lower right")

    fig.suptitle(
        "RFB–HORN for compact spoken-digit recognition",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(OUT / "fig_hero.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_hero.png", bbox_inches="tight")
    plt.close()
    print(f"wrote {OUT / 'fig_hero.pdf'}")


if __name__ == "__main__":
    main()
