"""
ResonanceDB (self-contained demo)
=================================

This script demonstrates a phase-aware similarity store based on resonance
between complex waveforms ψ(x) = A(x) * exp(i * φ(x)). It:

- Implements core resonance primitives (polar/complex mapping, score S in [0,1]).
- Provides an ANN-friendly packing (unit-energy, 2L real vector) for fast ranking.
- Builds a synthetic dataset with semantic operators (NEG, SHIFT+, INT_UP, INT_DOWN).
- Benchmarks operator-aware retrieval (P@1) vs a cosine-on-amplitude baseline.
- Includes a simple in-memory exact scanner and optional approximate path.
- Inspired by https://arxiv.org/abs/2509.09691

Run:
  python examples/resonanceDB.py --num_bases 200 --L 512 --seed 42

Export/import with MNIST autoencoder:
  1) Export complex encoder states to NPZ from the MNIST example:
     >>> from examples.image_mnist_oscillatory_autoencoder import export_encoder_complex_states
     >>> export_encoder_complex_states(model, images[:1000], "outputs/mnist_complex_states.npz")

  2) Load and index the NPZ here, then query:
     $ python examples/resonanceDB.py --load_npz outputs/mnist_complex_states.npz

This is self-contained but integrates smoothly with oscnet. A small bridge is
provided to convert HORN cell states (x, v, ω) into complex form.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ========= Resonance primitives =========

def polar_to_complex(amplitude: jnp.ndarray, phase: jnp.ndarray) -> jnp.ndarray:
    """Return complex vector z = A * exp(i * φ).

    Shapes: amplitude, phase: [..., L] -> z: [..., L] complex64/complex128
    """
    return amplitude * jnp.exp(1j * phase)


def complex_to_polar(z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return amplitude and phase from complex vector z.

    amplitude = |z| >= 0, phase = angle(z) in [-π, π)
    """
    return jnp.abs(z), jnp.angle(z)


def sign_phase_map(v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Map real vector v to polar form: A=|v|, φ = 0 if v>=0 else π.

    This preserves sign information in phase while remaining drop-in compatible
    with existing real embeddings.
    """
    amplitude = jnp.abs(v)
    phase = jnp.where(v >= 0.0, 0.0, jnp.pi)
    return amplitude, phase


def resonance_score(z1: jnp.ndarray, z2: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute resonance similarity S(z1, z2) in [0, 1].

    z1, z2: [..., L] complex arrays. Broadcasts leading dims.

    S = ((E1 + E2 + 2*Re<z1, z2>) * sqrt(E1*E2)) / (E1 + E2)^2
    with convention S=0 if E1+E2=0.
    """
    # Energies along last axis
    E1 = jnp.sum(jnp.abs(z1) ** 2, axis=-1)
    E2 = jnp.sum(jnp.abs(z2) ** 2, axis=-1)
    # vdot conjugates first arg when 1D; for ND we compute elementwise then reduce
    I = jnp.sum(jnp.real(z1 * jnp.conj(z2)), axis=-1)

    denom = (E1 + E2) ** 2
    safe = denom > eps
    numer = (E1 + E2 + 2.0 * I) * jnp.sqrt(jnp.maximum(E1 * E2, 0.0))
    S = jnp.where(safe, numer / (denom + (~safe) * eps), 0.0)
    # Clamp minor numerical drift
    return jnp.clip(S, 0.0, 1.0)


def pack_unit_2L(z: jnp.ndarray, eps: float = 1e-8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pack complex vector to 2L real unit vector for ANN ranking and return sqrt(E).

    Returns (u, sqrtE):
      - u: [..., 2L] real unit vector (concatenate real and imag parts after L2-norm)
      - sqrtE: [...,] scalar sqrt(E) so scale can be recovered if needed
    """
    E = jnp.sum(jnp.abs(z) ** 2, axis=-1)
    inv_norm = 1.0 / jnp.sqrt(E + eps)
    zt = z * jnp.expand_dims(inv_norm, axis=-1)
    u = jnp.concatenate([jnp.real(zt), jnp.imag(zt)], axis=-1)
    return u, jnp.sqrt(E + eps)


def cosine_similarity(u: jnp.ndarray, v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Cosine similarity on real vectors along last axis."""
    num = jnp.sum(u * v, axis=-1)
    den = jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1) + eps
    return num / den


# ========= Simple in-memory Resonance store =========

@dataclass
class ResonanceItem:
    key: str
    amplitude: jnp.ndarray  # [L]
    phase: jnp.ndarray      # [L]
    energy: float
    packed_unit: Optional[jnp.ndarray] = None  # [2L] (optional ANN-friendly)


class ResonanceMemory:
    """Minimal in-memory resonance store with exact scan and ANN-friendly packing."""

    def __init__(self, vector_length: int):
        self.L = vector_length
        self._items: List[ResonanceItem] = []
        self._packed_matrix: Optional[jnp.ndarray] = None  # [N, 2L]

    def add(self, key: str, amplitude: jnp.ndarray, phase: jnp.ndarray) -> None:
        assert amplitude.shape[-1] == self.L and phase.shape[-1] == self.L
        z = polar_to_complex(amplitude, phase)
        energy = float(jnp.sum(jnp.abs(z) ** 2))
        self._items.append(ResonanceItem(key=key, amplitude=amplitude, phase=phase, energy=energy))
        self._packed_matrix = None  # invalidate

    def __len__(self) -> int:
        return len(self._items)

    def build_ann_packing(self) -> None:
        if not self._items:
            self._packed_matrix = jnp.zeros((0, 2 * self.L), dtype=jnp.float32)
            return
        z_all = jnp.stack([polar_to_complex(it.amplitude, it.phase) for it in self._items], axis=0)
        u_all, _ = pack_unit_2L(z_all)
        self._packed_matrix = u_all.astype(jnp.float32)
        # Save per-item packed vectors for convenience
        for i, it in enumerate(self._items):
            self._items[i].packed_unit = self._packed_matrix[i]

    def query_exact(self, amplitude_q: jnp.ndarray, phase_q: jnp.ndarray, top_k: int = 10,
                    exclude_key: Optional[str] = None) -> List[Tuple[str, float]]:
        """Exact scan using full resonance score."""
        if len(self._items) == 0:
            return []
        zq = polar_to_complex(amplitude_q, phase_q)
        # Stack memory
        z_mem = jnp.stack([polar_to_complex(it.amplitude, it.phase) for it in self._items], axis=0)
        S = resonance_score(z_mem, jnp.expand_dims(zq, axis=0))  # [N]
        scores = list(zip([it.key for it in self._items], list(map(float, S))))
        if exclude_key is not None:
            scores = [kv for kv in scores if kv[0] != exclude_key]
        scores.sort(key=lambda kv: kv[1], reverse=True)
        return scores[:top_k]

    def query_ann_style(self, amplitude_q: jnp.ndarray, phase_q: jnp.ndarray, top_k: int = 10,
                        refine_exact: bool = True, exclude_key: Optional[str] = None) -> List[Tuple[str, float]]:
        """ANN-friendly: rank by dot in 2L space, optionally refine top_k with exact S.

        Note: This is a CPU in-memory dot; to scale, store self._packed_matrix in a real ANN index.
        """
        if self._packed_matrix is None:
            self.build_ann_packing()
        if self._packed_matrix.shape[0] == 0:
            return []
        zq = polar_to_complex(amplitude_q, phase_q)
        uq, _ = pack_unit_2L(jnp.expand_dims(zq, axis=0))  # [1, 2L]
        sims = jnp.squeeze(self._packed_matrix @ uq.T, axis=1)  # [N]
        # higher dot -> higher resonance in normalized space
        idx = jnp.argsort(-sims)[: (top_k * 4 if refine_exact else top_k)]
        candidates = [(self._items[int(i)].key, float(sims[int(i)])) for i in idx]
        if exclude_key is not None:
            candidates = [kv for kv in candidates if kv[0] != exclude_key]
        if not refine_exact:
            return candidates[:top_k]
        # Refine with exact S
        top_keys = [k for k, _ in candidates[: (top_k * 2)]]
        amp = jnp.stack([self._items[self._key_to_index(k)].amplitude for k in top_keys], axis=0)
        phs = jnp.stack([self._items[self._key_to_index(k)].phase for k in top_keys], axis=0)
        zc = polar_to_complex(amp, phs)
        Sq = resonance_score(zc, jnp.expand_dims(zq, axis=0))
        refined = list(zip(top_keys, list(map(float, Sq))))
        refined.sort(key=lambda kv: kv[1], reverse=True)
        return refined[:top_k]

    def _key_to_index(self, key: str) -> int:
        for i, it in enumerate(self._items):
            if it.key == key:
                return i
        raise KeyError(key)

    @staticmethod
    def from_npz(npz_path: str) -> "ResonanceMemory":
        """Load memory from NPZ with keys 'amplitude' (N,L), 'phase' (N,L)."""
        data = np.load(npz_path)
        amp = jnp.asarray(data["amplitude"])  # [N, L]
        phs = jnp.asarray(data["phase"])      # [N, L]
        N, L = amp.shape
        mem = ResonanceMemory(vector_length=L)
        for i in range(N):
            mem.add(key=f"npz:{i}", amplitude=amp[i], phase=phs[i])
        mem.build_ann_packing()
        return mem


# ========= Synthetic operator dataset =========

OPERATORS = ("BASE", "NEG", "SHIFT+", "INT_UP", "INT_DOWN")


@dataclass
class LabeledPattern:
    base_id: int
    operator: str
    amplitude: jnp.ndarray  # [L]
    phase: jnp.ndarray      # [L]
    key: str


def generate_bases(num_bases: int, L: int, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate base amplitude and zero-phase patterns.

    Amplitudes are positive random (|N(0,1)| smoothed). Phases initialized to zero.
    """
    k1, _ = jax.random.split(key)
    amp = jnp.abs(jax.random.normal(k1, (num_bases, L)).astype(jnp.float32))
    # Slightly smooth amplitudes to reduce pathology
    if L >= 4:
        kernel = jnp.array([0.25, 0.5, 0.25], dtype=amp.dtype)
        pad = jnp.pad(amp, ((0, 0), (1, 1)))
        amp = 0.5 * amp + 0.5 * (
            kernel[0] * pad[:, :-2] + kernel[1] * pad[:, 1:-1] + kernel[2] * pad[:, 2:]
        )
    phase = jnp.zeros((num_bases, L), dtype=jnp.float32)
    return amp, phase


def apply_operator(amplitude: jnp.ndarray, phase: jnp.ndarray, operator: str,
                   shift_delta: float, int_up: float, int_down: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if operator == "BASE":
        return amplitude, phase
    if operator == "NEG":
        return amplitude, (phase + jnp.pi)  # anti-phase
    if operator == "SHIFT+":
        return amplitude, (phase + shift_delta)
    if operator == "INT_UP":
        return amplitude * int_up, phase
    if operator == "INT_DOWN":
        return amplitude * int_down, phase
    raise ValueError(operator)


def build_operator_corpus(num_bases: int, L: int, key: jax.Array,
                          shift_delta: float = 0.6, int_up: float = 1.5, int_down: float = 0.67
                          ) -> List[LabeledPattern]:
    base_amp, base_phase = generate_bases(num_bases, L, key)
    corpus: List[LabeledPattern] = []
    for i in range(num_bases):
        for op in OPERATORS[1:]:  # exclude BASE to avoid trivial duplicates
            amp_i, ph_i = apply_operator(base_amp[i], base_phase[i], op, shift_delta, int_up, int_down)
            key_i = f"base{i:05d}:{op}"
            corpus.append(LabeledPattern(base_id=i, operator=op, amplitude=amp_i, phase=ph_i, key=key_i))
    return corpus


# ========= Evaluation =========

@dataclass
class EvalResult:
    p_at_1_by_operator: Dict[str, float]
    mean_dist_by_operator: Dict[str, float]
    std_dist_by_operator: Dict[str, float]


def evaluate_operator_retrieval(corpus: List[LabeledPattern], L: int, top_k: int = 1,
                                use_resonance: bool = True) -> EvalResult:
    """Operator classification by cross-base nearest-neighbor.

    For each query (base i, operator op), search among all items with base != i.
    We consider the operator label of the top-1 match; success if it equals op.

    Similarity choice:
      - Resonance: S between complex z vectors.
      - Cosine baseline: cosine between amplitude-only vectors (phase ignored).
    """
    mem = ResonanceMemory(vector_length=L)
    for item in corpus:
        mem.add(item.key, item.amplitude, item.phase)
    mem.build_ann_packing()  # precompute

    # Organize items for quick access
    keys = [it.key for it in mem._items]
    base_ids = jnp.array([lp.base_id for lp in corpus])
    ops = [lp.operator for lp in corpus]
    amps = jnp.stack([lp.amplitude for lp in corpus], axis=0)
    phs = jnp.stack([lp.phase for lp in corpus], axis=0)
    z_all = polar_to_complex(amps, phs)

    successes: Dict[str, int] = {op: 0 for op in OPERATORS[1:]}
    totals: Dict[str, int] = {op: 0 for op in OPERATORS[1:]}
    dists_mean: Dict[str, float] = {}
    dists_std: Dict[str, float] = {}

    # Precompute normalized amplitude for cosine baseline
    amp_norm = amps / (jnp.linalg.norm(amps, axis=1, keepdims=True) + 1e-8)

    for idx_q, q in enumerate(corpus):
        totals[q.operator] += 1
        mask = base_ids != q.base_id
        cand_idx = jnp.where(mask)[0]
        if use_resonance:
            zq = z_all[idx_q]
            S = resonance_score(z_all[cand_idx], jnp.expand_dims(zq, axis=0))
            # distance on [0,1]
            d = 1.0 - S
        else:
            uq = amp_norm[idx_q]
            sims = jnp.sum(amp_norm[cand_idx] * uq, axis=1)
            # Map cosine in [-1,1] to distance in [0,1]
            d = 0.5 * (1.0 - sims)
        # Top-1 neighbor
        top = int(cand_idx[jnp.argmin(d)])
        if ops[top] == q.operator:
            successes[q.operator] += 1

    for op in OPERATORS[1:]:
        p1 = successes[op] / max(totals[op], 1)
        # Collect distances distribution for reporting
        ds: List[float] = []
        for idx_q, q in enumerate(corpus):
            if q.operator != op:
                continue
            mask = base_ids != q.base_id
            cand_idx = jnp.where(mask)[0]
            if use_resonance:
                zq = z_all[idx_q]
                S = resonance_score(z_all[cand_idx], jnp.expand_dims(zq, axis=0))
                di = 1.0 - S
            else:
                uq = amp_norm[idx_q]
                sims = jnp.sum(amp_norm[cand_idx] * uq, axis=1)
                di = 0.5 * (1.0 - sims)
            ds.extend(list(map(float, di)))
        mean_d = float(jnp.mean(jnp.array(ds))) if ds else 0.0
        std_d = float(jnp.std(jnp.array(ds))) if ds else 0.0
        dists_mean[op] = mean_d
        dists_std[op] = std_d

    return EvalResult(
        p_at_1_by_operator={op: successes[op] / max(totals[op], 1) for op in OPERATORS[1:]},
        mean_dist_by_operator=dists_mean,
        std_dist_by_operator=dists_std,
    )


# ========= Optional oscnet bridge =========

def x_v_omega_to_complex(x: jnp.ndarray, v: jnp.ndarray, omega: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Convert amplitude-velocity states to complex: z = x + i * (v / ω).

    Shapes: x, v: [..., L], omega: [L] or [..., L].
    """
    c = 1.0 / (omega + eps)
    return x + 1j * (c * v)


# ========= CLI / Demo =========

def run_demo(args: argparse.Namespace) -> None:
    print("=== ResonanceDB Demo ===")
    print(f"Config: bases={args.num_bases}, L={args.L}, shift={args.shift_delta}, int_up={args.int_up}, int_down={args.int_down}")
    key = jax.random.PRNGKey(args.seed)

    # Build corpus
    t0 = time.time()
    corpus = build_operator_corpus(
        num_bases=args.num_bases,
        L=args.L,
        key=key,
        shift_delta=args.shift_delta,
        int_up=args.int_up,
        int_down=args.int_down,
    )
    build_time = time.time() - t0
    print(f"Built corpus with {len(corpus)} items in {build_time:.3f}s")

    # Build memory
    mem = ResonanceMemory(vector_length=args.L)
    for lp in corpus:
        mem.add(lp.key, lp.amplitude, lp.phase)
    mem.build_ann_packing()

    # Evaluate resonance
    print("\n-- Evaluating resonance (operator-aware retrieval across bases) --")
    t0 = time.time()
    res_res = evaluate_operator_retrieval(corpus, args.L, top_k=1, use_resonance=True)
    t_res = time.time() - t0
    # Evaluate cosine baseline (amplitude only)
    print("-- Evaluating cosine baseline (amplitude-only) --")
    t0 = time.time()
    cos_res = evaluate_operator_retrieval(corpus, args.L, top_k=1, use_resonance=False)
    t_cos = time.time() - t0

    # Report
    def fmt_dict(d: Dict[str, float]) -> str:
        return ", ".join([f"{k}: {v:.3f}" for k, v in d.items()])

    print("\nP@1 (Resonance):    ", fmt_dict(res_res.p_at_1_by_operator))
    print("P@1 (Cosine base): ", fmt_dict(cos_res.p_at_1_by_operator))
    print("\nMean distance by operator (lower is closer):")
    print("Resonance mean d:  ", fmt_dict(res_res.mean_dist_by_operator))
    print("Cosine mean d:     ", fmt_dict(cos_res.mean_dist_by_operator))
    print("\nResonance eval time: {:.3f}s | Cosine eval time: {:.3f}s".format(t_res, t_cos))

    # Quick functional test of exact vs ANN-style ranking equivalence (normalized)
    q = corpus[0]
    exact = mem.query_exact(q.amplitude, q.phase, top_k=5, exclude_key=q.key)
    ann = mem.query_ann_style(q.amplitude, q.phase, top_k=5, refine_exact=True, exclude_key=q.key)
    print("\nQuery sample:")
    print(" exact top-5:", exact)
    print(" ann   top-5:", ann)


def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResonanceDB Demo")
    p.add_argument("--num_bases", type=int, default=200)
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--shift_delta", type=float, default=0.6)
    p.add_argument("--int_up", type=float, default=1.5)
    p.add_argument("--int_down", type=float, default=0.67)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load_npz", type=str, default=None, help="Optional NPZ file with amplitude/phase to index")
    return p.parse_args()


if __name__ == "__main__":
    args = build_argparser()
    if args.load_npz:
        # Simple index-and-query path from NPZ
        mem = ResonanceMemory.from_npz(args.load_npz)
        print(f"Loaded NPZ with {len(mem)} items. Built ANN packing.")
        # Make a quick self-query to demonstrate API
        it0 = mem._items[0]
        exact = mem.query_exact(it0.amplitude, it0.phase, top_k=5, exclude_key=it0.key)
        ann = mem.query_ann_style(it0.amplitude, it0.phase, top_k=5, refine_exact=True, exclude_key=it0.key)
        print("Exact:", exact)
        print("ANN  :", ann)
    else:
        run_demo(args)


