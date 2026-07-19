"""Intuitive architecture schematic for an oscillatory trace.

Renders an SVG story of the setup: drive → coupled oscillator field →
decoder / readout. Oscillators are drawn as sites; sparse or top-k coupling
edges show who talks to whom.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np

from ..geometry import (
    infer_grid_shape,
    oscillator_pack_shape,
    oscillator_site_positions,
    select_batch,
    trajectory_keyframes,
)
from ..schema import GridShape, TraceBundle
from .base import ViewContext, ViewResult

Array = np.ndarray


class ArchitectureView:
    name = "architecture"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        svg = render_architecture_svg(bundle, batch_index=context.batch_index)
        path = context.output_dir / "architecture.svg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        return ViewResult(name=self.name, paths=[path])


def site_phase_series(
    bundle: TraceBundle,
    *,
    batch_index: int = 0,
    max_frames: int = 32,
) -> Optional[dict]:
    """Return keyframed per-site state for animating / inspecting nodes.

    Always includes ``phases`` / ``position`` with shape ``(T, N)``. When a
    velocity trajectory exists (HORN), also includes ``velocity`` and
    harmonic-style ``energy``.
    """

    if bundle.phase_trajectory is None:
        return None
    position = _site_time_series(bundle.phase_trajectory, batch_index=batch_index)
    if position is None:
        return None

    indices = trajectory_keyframes(position.shape[0], max_frames=max_frames)
    if not indices:
        return None
    index_list = list(indices)
    phases = np.asarray(position[index_list], dtype=np.float64)
    order = np.abs(np.mean(np.exp(1j * phases), axis=1))
    omegas = _site_omegas(bundle)
    omega_list = None
    omega_vec = None
    if omegas is not None:
        omega_vec = np.asarray(omegas, dtype=np.float64).reshape(-1)[: phases.shape[1]]
        omega_list = [float(v) for v in omega_vec]

    velocity = None
    energy = None
    if bundle.velocity_trajectory is not None:
        vel_series = _site_time_series(
            bundle.velocity_trajectory, batch_index=batch_index
        )
        if vel_series is not None and vel_series.shape[-1] == phases.shape[1]:
            velocity = np.asarray(vel_series[index_list], dtype=np.float64)
            if omega_vec is None:
                omega_vec = np.ones(phases.shape[1], dtype=np.float64)
            # Harmonic oscillator energy: ½ (ω² x² + v²)
            energy = 0.5 * (
                (omega_vec[None, :] ** 2) * (phases**2) + (velocity**2)
            )

    payload = {
        "labels": [f"t={index}" for index in index_list],
        "step_indices": [int(i) for i in index_list],
        "phases": phases.tolist(),
        "position": phases.tolist(),
        "order_parameter": [float(v) for v in order],
        "omega": omega_list,
        "n_sites": int(phases.shape[1]),
        "has_velocity": velocity is not None,
    }
    if velocity is not None:
        payload["velocity"] = velocity.tolist()
        payload["energy"] = np.asarray(energy, dtype=np.float64).tolist()
    return payload


def _site_time_series(
    trajectory: Array,
    *,
    batch_index: int,
) -> Optional[Array]:
    """Reduce a trajectory to ``(T, sites)`` for one batch item."""

    theta = np.asarray(trajectory)
    if theta.ndim == 3:
        return select_batch(theta, batch_index, batch_axis=1)
    if theta.ndim == 4:
        selected = select_batch(theta, batch_index, batch_axis=1)
        return selected[..., 0]
    if theta.ndim == 5:
        selected = select_batch(theta, batch_index, batch_axis=1)
        return selected[..., 0].reshape(selected.shape[0], -1)
    return None


def render_architecture_svg(
    bundle: TraceBundle,
    *,
    batch_index: int = 0,
    max_edges: int = 180,
) -> str:
    """Return a self-contained SVG schematic for ``bundle``."""

    n_sites = _n_sites(bundle)
    grid = _schematic_grid(bundle, n_sites)
    nodes = _node_positions(n_sites, grid)
    edge_budget = _edge_budget(n_sites, max_edges=max_edges)
    # Pull a larger candidate pool so local filtering still has choices.
    candidates = _coupling_edges(bundle, max_edges=max(edge_budget * 8, edge_budget))
    has_coupling = _has_coupling_matrix(bundle)
    omegas = _site_omegas(bundle)
    phases = _site_phases(bundle, batch_index=batch_index)

    width, height = 980, 440
    field_x, field_y, field_w, field_h = 250, 48, 480, 320
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="OscNet architecture schematic">',
        _styles(),
    ]

    # Pipeline blocks
    parts.append(_block(24, 138, 160, 140, "drive", _drive_label(bundle)))
    parts.append(_arrow(184, 208, 250, 208))
    parts.append(
        f'<rect class="field" x="{field_x}" y="{field_y}" width="{field_w}" '
        f'height="{field_h}" rx="18" />'
    )
    parts.append(
        f'<text class="field-label" x="{field_x + 16}" y="{field_y + 28}">'
        f"oscillator field</text>"
    )

    drawn_edges: List[Tuple[int, int, float]] = []

    # Map unit positions into the field card.
    if nodes:
        xs = [p[0] for p in nodes]
        ys = [p[1] for p in nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 36
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)

        def map_point(x: float, y: float) -> Tuple[float, float]:
            px = field_x + pad + (x - min_x) / span_x * (field_w - 2 * pad)
            py = field_y + pad + 8 + (y - min_y) / span_y * (field_h - 2 * pad - 8)
            return px, py

        mapped = [map_point(x, y) for x, y in nodes]
        radius = _node_radius(n_sites, grid)
        drawn_edges = _select_drawable_edges(
            candidates,
            mapped,
            max_edges=edge_budget,
        )

        # Edges under nodes.
        if drawn_edges:
            max_w = max(abs(w) for _, _, w in drawn_edges) or 1.0
            # Large fields stay readable with lighter strokes.
            opacity_scale = 0.75 if n_sites > 128 else 1.0
            for i, j, w in drawn_edges:
                x1, y1 = mapped[i]
                x2, y2 = mapped[j]
                strength = abs(w) / max_w
                opacity = (0.14 + 0.50 * strength) * opacity_scale
                width_px = 0.55 + 1.6 * strength
                cls = "edge-pos" if w >= 0 else "edge-neg"
                parts.append(
                    f'<line class="osc-edge {cls}" data-src="{i}" data-dst="{j}" '
                    f'data-w="{w:.6g}" data-strength="{strength:.4f}" '
                    f'x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                    f'stroke-width="{width_px:.2f}" opacity="{opacity:.3f}">'
                    f"<title>coupling {i}–{j}: {w:.4g}</title></line>"
                )

        for index, (px, py) in enumerate(mapped):
            fill = _node_fill(index, omegas, phases)
            omega_txt = ""
            if omegas is not None and index < len(np.asarray(omegas).reshape(-1)):
                omega_txt = f" ω={float(np.asarray(omegas).reshape(-1)[index]):.3g}"
            parts.append(
                f'<circle class="node osc-node" data-site="{index}" data-r="{radius:.2f}" '
                f'cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{fill}" '
                f'style="cursor:pointer">'
                f"<title>site {index}{escape(omega_txt)}</title></circle>"
            )
    else:
        parts.append(
            f'<text class="empty" x="{field_x + field_w / 2}" y="{field_y + field_h / 2}" '
            f'text-anchor="middle">no oscillator sites in trace</text>'
        )

    parts.append(_arrow(730, 208, 790, 208))
    parts.append(_block(790, 138, 160, 140, "decode", _decode_label(bundle)))

    caption = _caption(
        bundle,
        n_sites,
        grid,
        drawn_edges=drawn_edges,
        has_coupling=has_coupling,
    )
    parts.append(_legend(24, 388, caption=caption))
    parts.append("</svg>")
    return "\n".join(parts)


def _styles() -> str:
    return """
<style>
  .field { fill: #f3f1ea; stroke: #cfc9bb; stroke-width: 1.5; }
  .field-label { font: 600 12px ui-sans-serif, system-ui, sans-serif; fill: #5c5955; }
  .block { fill: #ffffff; stroke: #1c1b19; stroke-width: 1.5; }
  .block-title { font: 700 13px ui-sans-serif, system-ui, sans-serif; fill: #1c1b19; }
  .block-body { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #5c5955; }
  .arrow { stroke: #1c1b19; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
  .node { stroke: #1c1b19; stroke-width: 0.8; }
  .node.is-selected { stroke: #1c1b19; stroke-width: 2.2; }
  .node.is-dimmed { opacity: 0.22; }
  .edge-pos { stroke: #b42318; }
  .edge-neg { stroke: #175cd3; }
  .osc-edge.is-dimmed { opacity: 0.04 !important; }
  .osc-edge.is-hot { opacity: 0.95 !important; stroke-width: 2.8; }
  .legend { font: 11px ui-sans-serif, system-ui, sans-serif; fill: #5c5955; }
  .empty { font: 13px ui-sans-serif, system-ui, sans-serif; fill: #5c5955; }
</style>
<defs>
  <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L6,3 L0,6 Z" fill="#1c1b19" />
  </marker>
</defs>
"""


def _block(x: float, y: float, w: float, h: float, title: str, body: str) -> str:
    lines = body.split("\n")
    text = [
        f'<rect class="block" x="{x}" y="{y}" width="{w}" height="{h}" rx="14" />',
        f'<text class="block-title" x="{x + 16}" y="{y + 32}">{escape(title)}</text>',
    ]
    for i, line in enumerate(lines):
        text.append(
            f'<text class="block-body" x="{x + 16}" y="{y + 58 + 18 * i}">{escape(line)}</text>'
        )
    return "\n".join(text)


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<line class="arrow" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />'


def _legend(x: float, y: float, *, caption: str = "") -> str:
    cap = (
        f'<text class="legend" x="720" y="4" text-anchor="end">{escape(caption)}</text>'
        if caption
        else ""
    )
    return f"""
<g transform="translate({x},{y})">
  <circle class="node" cx="8" cy="0" r="5" fill="rgb(180,35,24)" />
  <circle class="node" cx="22" cy="0" r="5" fill="rgb(245,245,245)" />
  <circle class="node" cx="36" cy="0" r="5" fill="rgb(23,92,211)" />
  <text class="legend" x="48" y="4">Δphase from t=0 (red / none / blue)</text>
  <line class="edge-pos" x1="280" y1="0" x2="310" y2="0" stroke-width="2" />
  <text class="legend" x="318" y="4">excitatory</text>
  <line class="edge-neg" x1="400" y1="0" x2="430" y2="0" stroke-width="2" />
  {cap}
</g>
"""


def _caption(
    bundle: TraceBundle,
    n_sites: int,
    grid: Optional[GridShape],
    *,
    drawn_edges: Sequence[Tuple[int, int, float]],
    has_coupling: bool,
) -> str:
    """Compact layout note — only describe coupling that the diagram shows."""

    if grid is not None and grid[0] * grid[1] != n_sites:
        layout = f"{grid[0]}×{grid[1]} pack · {n_sites} sites"
    elif grid is not None:
        layout = f"{grid[0]}×{grid[1]}"
    else:
        layout = f"{n_sites} sites"
    if drawn_edges:
        return f"{layout} · {len(drawn_edges)} links"
    if has_coupling:
        return f"{layout} · coupling not drawn"
    if _uses_spatial_coupling(bundle):
        return f"{layout} · spatial / implicit"
    return layout


def _has_coupling_matrix(bundle: TraceBundle) -> bool:
    for matrix in (bundle.coupling_profile, bundle.coupling):
        if matrix is None:
            continue
        arr = np.asarray(matrix)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size:
            return True
    return False


def _uses_spatial_coupling(bundle: TraceBundle) -> bool:
    if bundle.family in {"winfree", "phase_flow"}:
        return True
    theta = bundle.phase_trajectory
    return theta is not None and np.asarray(theta).ndim == 5


def _edge_budget(n_sites: int, *, max_edges: int) -> int:
    if n_sites <= 64:
        return max_edges
    if n_sites <= 256:
        return min(max_edges, 140)
    return min(max_edges, 220)


def _select_drawable_edges(
    edges: Sequence[Tuple[int, int, float]],
    mapped: Sequence[Tuple[float, float]],
    *,
    max_edges: int,
) -> List[Tuple[int, int, float]]:
    """Choose edges to draw without inventing a false topology.

    Moderate banks (≤128): keep strongest edges as-is.

    Large banks: prefer short geometric hops so dense matrices do not
    become an unreadable hairball.
    """

    if not edges or not mapped:
        return []
    if len(mapped) <= 128:
        return list(edges[:max_edges])

    scored: List[Tuple[float, float, int, int, float]] = []
    for i, j, w in edges:
        if i >= len(mapped) or j >= len(mapped):
            continue
        x1, y1 = mapped[i]
        x2, y2 = mapped[j]
        dist = math.hypot(x2 - x1, y2 - y1)
        scored.append((dist, -abs(w), i, j, w))
    if not scored:
        return []
    scored.sort()
    # Keep edges near the local lattice scale (short hops first).
    nearest = scored[0][0]
    local_limit = max(nearest * 2.8, 1e-6)
    local = [(i, j, w) for dist, _, i, j, w in scored if dist <= local_limit]
    if len(local) >= max(12, max_edges // 4):
        return local[:max_edges]
    # Dense long-range fallback: strongest overall, fewer strokes.
    by_strength = sorted(edges, key=lambda item: abs(item[2]), reverse=True)
    return list(by_strength[: min(max_edges, 80)])


def _drive_label(bundle: TraceBundle) -> str:
    if bundle.family == "generator":
        if bundle.meta.get("has_condition_bank"):
            return "class / condition\nbank → field"
        return "state prior /\nlabel condition"
    if bundle.family == "winfree":
        return "encoder ω drive\n→ phase field"
    if bundle.family == "phase_flow":
        return "image / noise\n→ rate-phase field"
    return "input drive"


def _decode_label(bundle: TraceBundle) -> str:
    if bundle.readout is not None:
        shape = tuple(np.asarray(bundle.readout).shape)
        return f"decoder\n→ readout\n{shape}"
    if bundle.family == "winfree":
        return "latent /\nreconstruction"
    return "readout"


def _n_sites(bundle: TraceBundle) -> int:
    if bundle.meta.get("n_sites") is not None:
        return int(bundle.meta["n_sites"])
    if bundle.phase_trajectory is None:
        return 0
    theta = np.asarray(bundle.phase_trajectory)
    if theta.ndim == 5:
        return int(theta.shape[2] * theta.shape[3])
    if theta.ndim == 4:
        return int(theta.shape[-2])
    if theta.ndim == 3:
        return int(theta.shape[-1])
    return 0


def _schematic_grid(bundle: TraceBundle, n_sites: int) -> Optional[GridShape]:
    if bundle.grid_shape is not None:
        return bundle.grid_shape
    if bundle.phase_trajectory is not None and np.asarray(bundle.phase_trajectory).ndim == 5:
        theta = np.asarray(bundle.phase_trajectory)
        return (int(theta.shape[2]), int(theta.shape[3]))
    square = infer_grid_shape(n_sites)
    if square is not None:
        return square
    # Local coupling profiles use the model near-square pack (may have holes).
    # Dense banks stay on a circle so we don't fake a lattice.
    if n_sites > 0 and _looks_local_coupling(bundle):
        return oscillator_pack_shape(n_sites)
    return None


def _looks_local_coupling(bundle: TraceBundle) -> bool:
    matrix = bundle.coupling_profile
    if matrix is None:
        matrix = bundle.coupling
    if matrix is None:
        return False
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    n = matrix.shape[0]
    nnz = int(np.count_nonzero(np.abs(matrix) > 1e-8))
    # Sparse relative to dense N² (excluding diagonal).
    return nnz > 0 and nnz <= max(3 * n, 0.15 * n * n)


def _node_positions(
    n_sites: int,
    grid: Optional[GridShape],
) -> List[Tuple[float, float]]:
    if n_sites <= 0:
        return []
    if grid is not None and grid[0] * grid[1] == n_sites:
        height, width = grid
        return [(float(c), float(r)) for r in range(height) for c in range(width)]
    if grid is not None and grid[0] * grid[1] > n_sites:
        # Incomplete pack: sit sites on the same coordinates the model uses.
        return oscillator_site_positions(n_sites)
    # Circular fallback for unstructured banks.
    return [
        (math.cos(2 * math.pi * i / n_sites), math.sin(2 * math.pi * i / n_sites))
        for i in range(n_sites)
    ]


def _node_radius(n_sites: int, grid: Optional[GridShape]) -> float:
    if grid is not None:
        return max(3.0, min(10.0, 140.0 / max(grid)))
    if n_sites <= 24:
        return 9.0
    if n_sites <= 64:
        return 6.0
    return 4.0


def _coupling_edges(
    bundle: TraceBundle,
    *,
    max_edges: int,
) -> List[Tuple[int, int, float]]:
    matrix = None
    if bundle.coupling_profile is not None and np.asarray(bundle.coupling_profile).ndim == 2:
        matrix = np.asarray(bundle.coupling_profile, dtype=np.float64)
        # Prefer profile when it encodes topology; fall back to weights if dense ones.
        if np.count_nonzero(np.abs(matrix) > 1e-8) == matrix.size:
            matrix = None
    if matrix is None and bundle.coupling is not None and np.asarray(bundle.coupling).ndim == 2:
        matrix = np.asarray(bundle.coupling, dtype=np.float64)
    if matrix is None:
        return []

    n = matrix.shape[0]
    candidates: List[Tuple[float, int, int, float]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(matrix[i, j])
            if abs(w) <= 1e-8:
                continue
            # Undirected dedupe for drawing: keep stronger direction.
            if j < i:
                continue
            w_ji = float(matrix[j, i]) if j < n else 0.0
            w_draw = w if abs(w) >= abs(w_ji) else w_ji
            candidates.append((abs(w_draw), i, j, w_draw))
    candidates.sort(reverse=True)
    return [(i, j, w) for _, i, j, w in candidates[:max_edges]]


def _site_omegas(bundle: TraceBundle) -> Optional[Array]:
    if bundle.omega is None:
        return None
    omega = np.asarray(bundle.omega, dtype=np.float64)
    if omega.ndim == 0:
        return None
    if omega.ndim == 1:
        return omega
    # (B, sites, ...) or (sites, channels)
    if omega.ndim == 2:
        return omega.mean(axis=-1) if omega.shape[0] > omega.shape[1] else omega.mean(axis=0)
    if omega.ndim == 3:
        return omega.mean(axis=(0, -1))
    return omega.reshape(-1)[: _n_sites(bundle)]


def _site_phases(bundle: TraceBundle, *, batch_index: int) -> Optional[Array]:
    if bundle.phase_trajectory is None:
        return None
    theta = np.asarray(bundle.phase_trajectory)
    if theta.ndim == 3:
        # (T, B, sites)
        b = int(batch_index) % theta.shape[1]
        return theta[-1, b]
    if theta.ndim == 4:
        b = int(batch_index) % theta.shape[1]
        return theta[-1, b].mean(axis=-1)
    if theta.ndim == 5:
        b = int(batch_index) % theta.shape[1]
        return theta[-1, b].mean(axis=-1).reshape(-1)
    return None


def _node_fill(
    index: int,
    omegas: Optional[Array],
    phases: Optional[Array],
) -> str:
    # Prefer phase (intuitive "state of the medium"); else omega; else neutral.
    if phases is not None and index < len(phases):
        return _phase_color(float(phases[index]))
    if omegas is not None and index < len(omegas):
        return _omega_color(float(omegas[index]), omegas)
    return "#f97316"


def _phase_color(phase: float) -> str:
    # Twilight-ish hue around the circle.
    t = (phase % (2 * math.pi)) / (2 * math.pi)
    # Simple hue wheel without external deps.
    return _hsl(int(360 * t), 55, 58)


def _omega_color(omega: float, all_omegas: Array) -> str:
    lo = float(np.min(all_omegas))
    hi = float(np.max(all_omegas))
    t = 0.5 if hi <= lo else (omega - lo) / (hi - lo)
    # Orange → cream → blue
    r = int(244 - 120 * t)
    g = int(140 + 40 * (1 - abs(t - 0.5) * 2))
    b = int(60 + 160 * t)
    return f"rgb({r},{g},{b})"


def _hsl(h: int, s: int, l: int) -> str:
    return f"hsl({h} {s}% {l}%)"
