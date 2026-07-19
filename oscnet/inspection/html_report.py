"""Self-contained HTML report: one linked inspect stage + drawers."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from oscnet.analysis.phase_synchrony import circular_difference

from .schema import TraceBundle
from .views._plotting import imshow_phase, imshow_signed
from .views.architecture import site_phase_series
from .views.fields import extract_phase_frames

# Secondary surfaces shown as collapsible drawers (not primary tabs).
_DRAWER_ARTIFACTS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("coupling", "Coupling matrix", ("coupling.png",)),
    ("omega", "Frequency map", ("omega.png",)),
    ("synchrony", "Synchrony / energy plots", ("synchrony.png", "energy.png")),
    ("rate_fields", "Rate fields", ("rate_fields.png",)),
    ("vertical_gain", "Vertical gain", ("vertical_gain.png",)),
    ("readout", "Readout (decoder output)", ("readout.png",)),
)


def write_html_report(
    bundle: TraceBundle,
    output_dir: Path | str,
    *,
    artifacts: Sequence[str],
    skipped: Sequence[dict],
    batch_index: int = 0,
    max_frames: int = 32,
    dpi: int = 100,
) -> Path:
    """Write ``index.html`` as a unified linked inspect stage."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture_svg = None
    svg_path = output_dir / "architecture.svg"
    if svg_path.is_file():
        architecture_svg = svg_path.read_text()

    frame_budget = max(1, min(max_frames, 32))
    phase_series = site_phase_series(
        bundle,
        batch_index=batch_index,
        max_frames=frame_budget,
    )
    phase_frames = _phase_scrubber_payload(
        bundle,
        batch_index=batch_index,
        max_frames=frame_budget,
        dpi=dpi,
    )

    drawers: List[dict] = []
    for drawer_id, title, names in _DRAWER_ARTIFACTS:
        images = []
        for name in names:
            path = output_dir / name
            if path.is_file():
                images.append({"name": name, "src": _file_to_data_uri(path)})
        if images:
            drawers.append({"id": drawer_id, "title": title, "images": images})

    payload = {
        "family": bundle.family,
        "source": str(bundle.source),
        "batch_index": batch_index,
        "grid_shape": list(bundle.grid_shape) if bundle.grid_shape else None,
        "slots": [ref.summary() for ref in bundle.slots()],
        "raw_keys": [
            {"name": key, "shape": list(bundle.arrays[key].shape)}
            for key in bundle.keys
        ],
        "meta": dict(bundle.meta),
        "skipped": list(skipped),
        "artifacts": list(artifacts),
        "phase_series": phase_series,
        "stage": {
            "svg": architecture_svg,
            "frames": phase_frames or [],
        },
        "drawers": drawers,
    }
    path = output_dir / "index.html"
    path.write_text(_render_html(payload))
    return path


def _phase_scrubber_payload(
    bundle: TraceBundle,
    *,
    batch_index: int,
    max_frames: int,
    dpi: int,
) -> Optional[List[dict]]:
    if bundle.phase_trajectory is None:
        return None
    n_steps = int(np.asarray(bundle.phase_trajectory).shape[0])
    if n_steps <= 0:
        return None
    frames, titles = extract_phase_frames(
        bundle.phase_trajectory,
        batch_index=batch_index,
        grid_shape=bundle.grid_shape,
        max_frames=max(1, min(max_frames, n_steps)),
    )
    if not frames:
        return None
    base = np.asarray(frames[0], dtype=np.float64)
    deltas = [
        circular_difference(np.asarray(frame, dtype=np.float64), base) for frame in frames
    ]
    delta_limit = float(max((np.max(np.abs(delta)) for delta in deltas), default=0.0))
    delta_limit = max(delta_limit, 1e-3)
    encoded = []
    for frame, delta, title in zip(frames, deltas, titles):
        encoded.append(
            {
                "label": title,
                "src": _array_to_phase_scrubber_uri(
                    frame,
                    delta,
                    title=title,
                    delta_limit=delta_limit,
                    dpi=dpi,
                ),
            }
        )
    return encoded


def _array_to_phase_scrubber_uri(
    frame,
    delta,
    *,
    title: str,
    delta_limit: float,
    dpi: int,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    imshow_phase(axes[0], frame, title="phase")
    im = imshow_signed(axes[1], delta, title="Δ from t=0")
    im.set_clim(-delta_limit, delta_limit)
    fig.suptitle(title, fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=max(dpi, 120), bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _file_to_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _stats_row(payload: dict) -> str:
    series = payload.get("phase_series") or {}
    n_sites = series.get("n_sites") or (payload.get("meta") or {}).get("n_sites") or "—"
    n_steps = len(series.get("phases") or [])
    order = series.get("order_parameter") or []
    r0 = f"{order[0]:.3f}" if order else "—"
    r1 = f"{order[-1]:.3f}" if order else "—"
    return f"""
<div class="stats">
  <div><span class="k">family</span><span class="v">{html.escape(str(payload['family']))}</span></div>
  <div><span class="k">sites</span><span class="v">{html.escape(str(n_sites))}</span></div>
  <div><span class="k">settle frames</span><span class="v">{html.escape(str(n_steps))}</span></div>
  <div><span class="k">R(t0)→R(tend)</span><span class="v">{html.escape(f"{r0} → {r1}")}</span></div>
  <div><span class="k">batch</span><span class="v">{html.escape(str(payload.get('batch_index', 0)))}</span></div>
</div>
"""


def _multiples_html(frames: Sequence[dict]) -> str:
    if not frames:
        return ""
    buttons = []
    for index, frame in enumerate(frames):
        label = html.escape(str(frame.get("label", f"t={index}")))
        src = html.escape(frame["src"], quote=True)
        buttons.append(
            f'<button type="button" class="multiplicity" data-step="{index}">'
            f'<img src="{src}" alt="{label}" /><span>{label}</span></button>'
        )
    return f'<div class="multiples">{"".join(buttons)}</div>'


def _keys_drawer(payload: dict) -> str:
    grid = payload.get("grid_shape")
    grid_text = "×".join(str(v) for v in grid) if grid else "—"
    meta = payload.get("meta") or {}
    meta_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th>"
        f"<td><code>{html.escape(str(v))}</code></td></tr>"
        for k, v in sorted(meta.items(), key=lambda item: str(item[0]))
    )
    slot_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(slot['name'])}</code></td>"
        f"<td><code>{html.escape(str(tuple(slot['shape'])))}</code></td>"
        f"<td>{html.escape(slot.get('kind', ''))}</td></tr>"
        for slot in payload.get("slots") or []
    )
    key_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item['name'])}</code></td>"
        f"<td><code>{html.escape(str(tuple(item['shape'])))}</code></td></tr>"
        for item in payload.get("raw_keys") or []
    )
    return f"""
<details class="drawer">
  <summary>Trace keys &amp; metadata</summary>
  <table class="kv">
    <tr><th>source</th><td><code>{html.escape(str(payload['source']))}</code></td></tr>
    <tr><th>grid_shape</th><td><code>{html.escape(grid_text)}</code></td></tr>
    {meta_rows}
  </table>
  <h3>Normalized slots</h3>
  <table class="list">
    <thead><tr><th>slot</th><th>shape</th><th>kind</th></tr></thead>
    <tbody>{slot_rows}</tbody>
  </table>
  <h3>Raw keys</h3>
  <table class="list">
    <thead><tr><th>key</th><th>shape</th></tr></thead>
    <tbody>{key_rows}</tbody>
  </table>
</details>
"""


def _image_drawer(drawer: dict) -> str:
    imgs = "".join(
        f'<figure><img src="{html.escape(img["src"], quote=True)}" '
        f'alt="{html.escape(img["name"])}" />'
        f'<figcaption>{html.escape(img["name"])}</figcaption></figure>'
        for img in drawer["images"]
    )
    return (
        f'<details class="drawer" id="drawer-{html.escape(drawer["id"])}">'
        f'<summary>{html.escape(drawer["title"])}</summary>'
        f'<div class="gallery">{imgs}</div></details>'
    )


def _field_drawer(frames: Sequence[dict]) -> str:
    if not frames:
        return ""
    return f"""
<details class="drawer" id="drawer-fields">
  <summary>Phase field movie</summary>
  <div class="field-band">
    <div class="field-main">
      <img class="frame" alt="phase frame" />
    </div>
    <aside class="field-nav">
      {_multiples_html(frames)}
    </aside>
  </div>
</details>
"""


def _stage_html(payload: dict) -> str:
    stage = payload.get("stage") or {}
    svg = stage.get("svg") or ""
    schematic = f'<div class="schematic live-arch">{svg}</div>' if svg else ""

    return f"""
<div class="stage">
  {_stats_row(payload)}
  <div class="transport phase-scrub">
    <button type="button" class="play" aria-label="Play">Play</button>
    <input type="range" min="0" max="0" value="0" class="slider" aria-label="Time" />
    <div class="transport-readout">
      <span class="label">—</span>
      <span class="order-wrap"><span class="order-k">R</span> <span class="order">—</span></span>
      <svg class="order-spark" viewBox="0 0 120 24" width="120" height="24" aria-hidden="true"></svg>
    </div>
    <div class="transport-tools">
      <label class="tool">links <input type="range" class="edge-slider" min="0" max="100" value="35" /></label>
      <button type="button" class="clear-sel">Clear</button>
    </div>
  </div>
  <section class="arch-band">
    <div class="arch-main">
      <div class="band-label">architecture</div>
      {schematic}
    </div>
    <aside class="site-panel">
      <div class="band-label">site</div>
      <div class="site-info">Click an oscillator to inspect it.</div>
      <svg class="site-spark" viewBox="0 0 320 72" width="100%" height="72" hidden></svg>
      <div class="spark-legend" hidden>
        <span class="spark-x">phase</span>
        <span class="spark-v">v</span>
        <span class="spark-e">energy</span>
      </div>
    </aside>
  </section>
</div>
"""


def _render_html(payload: dict) -> str:
    data_json = json.dumps(payload).replace("<", "\\u003c")
    family = html.escape(str(payload["family"]))
    source = html.escape(str(payload["source"]))

    stage = payload.get("stage") or {}
    drawers_html = _field_drawer(stage.get("frames") or [])
    drawers_html += "".join(_image_drawer(d) for d in payload.get("drawers") or [])
    drawers_html += _keys_drawer(payload)

    skipped = payload.get("skipped") or []
    if skipped:
        items = "".join(
            f"<li><code>{html.escape(item.get('view', ''))}</code> — "
            f"{html.escape(item.get('reason', ''))}</li>"
            for item in skipped
        )
        skipped_html = (
            f'<details class="drawer skipped"><summary>Skipped views '
            f"({len(skipped)})</summary><ul>{items}</ul></details>"
        )
    else:
        skipped_html = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>OscNet inspect · {family}</title>
<style>
  :root {{
    --bg: #f7f6f3;
    --fg: #1c1b19;
    --muted: #5c5955;
    --line: #d9d5cc;
    --card: #ffffff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font: 15px/1.45 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    color: var(--fg);
    background: var(--bg);
  }}
  header {{
    padding: 16px clamp(16px, 2.5vw, 36px) 12px;
    background: var(--card);
    border-bottom: 1px solid var(--line);
  }}
  h1 {{
    margin: 0 0 4px;
    font-size: 20px;
    font-weight: 650;
    letter-spacing: -0.02em;
  }}
  h3 {{ margin: 16px 0 8px; font-size: 13px; }}
  .meta {{ color: var(--muted); font-size: 12px; word-break: break-all; }}
  main {{
    padding: 16px clamp(16px, 2.5vw, 36px) 48px;
    max-width: none;
    width: 100%;
  }}
  .band-label {{
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px;
    margin-bottom: 12px;
  }}
  .stats > div {{
    background: var(--card);
    border: 1px solid var(--line);
    padding: 10px 12px;
  }}
  .stats .k {{ display: block; color: var(--muted); font-size: 11px; margin-bottom: 2px; }}
  .stats .v {{ font-size: 14px; font-weight: 650; }}
  .transport {{
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    background: var(--card);
    border: 1px solid var(--line);
    padding: 10px 14px;
    margin-bottom: 16px;
  }}
  .transport .slider {{
    flex: 1 1 220px;
    min-width: 160px;
    accent-color: var(--fg);
  }}
  .transport-readout {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-variant-numeric: tabular-nums;
    font-size: 13px;
    color: var(--muted);
    white-space: nowrap;
  }}
  .transport-readout .label {{
    color: var(--fg);
    font-weight: 650;
    min-width: 4.5ch;
  }}
  .order-wrap {{ display: inline-flex; gap: 4px; align-items: baseline; }}
  .order-k {{ color: var(--muted); font-size: 11px; }}
  .order {{ color: var(--fg); font-weight: 650; }}
  .order-spark {{
    display: block;
    background: var(--bg);
    border: 1px solid var(--line);
  }}
  .transport-tools {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-left: auto;
    color: var(--muted);
    font-size: 13px;
  }}
  .edge-slider {{ width: 110px; vertical-align: middle; accent-color: var(--fg); }}
  .play, .clear-sel {{
    border: 1px solid var(--line);
    background: var(--bg);
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
  }}
  .play {{
    min-width: 4.5rem;
    background: var(--fg);
    color: #fff;
    border-color: var(--fg);
  }}
  .play:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .arch-band {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(260px, 320px);
    gap: 14px;
    margin-bottom: 16px;
    align-items: stretch;
  }}
  .field-band {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(220px, 280px);
    gap: 14px;
    margin-top: 12px;
    align-items: stretch;
  }}
  @media (max-width: 960px) {{
    .arch-band,
    .field-band {{ grid-template-columns: 1fr; }}
    .transport-tools {{ margin-left: 0; width: 100%; }}
  }}
  .arch-main,
  .site-panel {{
    background: var(--card);
    border: 1px solid var(--line);
    padding: 12px 14px;
  }}
  .field-main,
  .field-nav {{
    background: var(--bg);
    border: 1px solid var(--line);
    padding: 10px;
  }}
  .schematic {{
    min-height: min(52vh, 560px);
    display: flex;
    align-items: center;
  }}
  .schematic svg {{
    display: block;
    width: 100%;
    height: auto;
    max-height: min(58vh, 620px);
  }}
  .site-info {{ font-size: 13px; line-height: 1.4; }}
  .site-spark {{
    display: block;
    margin-top: 10px;
    background: var(--bg);
    border: 1px solid var(--line);
  }}
  .site-spark[hidden], .spark-legend[hidden] {{ display: none; }}
  .spark-legend {{
    display: flex;
    gap: 14px;
    margin-top: 4px;
    font-size: 11px;
    color: var(--muted);
  }}
  .spark-legend .spark-x::before,
  .spark-legend .spark-v::before,
  .spark-legend .spark-e::before {{
    content: "";
    display: inline-block;
    width: 10px;
    height: 2px;
    margin-right: 5px;
    vertical-align: middle;
    background: #1c1b19;
  }}
  .spark-legend .spark-v::before {{ background: #175cd3; }}
  .spark-legend .spark-e::before {{ background: #b42318; }}
  .frame {{
    display: block;
    width: 100%;
    height: auto;
    min-height: 280px;
    object-fit: contain;
    background: var(--bg);
    border: 1px solid var(--line);
  }}
  .multiples {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: min(70vh, 720px);
    overflow: auto;
  }}
  .multiplicity {{
    border: 1px solid var(--line);
    background: var(--bg);
    padding: 6px;
    cursor: pointer;
    width: 100%;
    text-align: left;
    font: inherit;
  }}
  .multiplicity img {{ display: block; width: 100%; height: auto; }}
  .multiplicity span {{ display: block; margin-top: 4px; font-size: 11px; color: var(--muted); }}
  .multiplicity.active {{ border-color: var(--fg); background: #fff; }}
  .drawer {{
    margin-top: 12px;
    background: var(--card);
    border: 1px solid var(--line);
    padding: 10px 12px;
  }}
  .drawer summary {{
    cursor: pointer;
    font-weight: 650;
    font-size: 14px;
  }}
  .drawer .gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
    margin-top: 12px;
  }}
  figure {{ margin: 0; }}
  figure img {{ display: block; max-width: 100%; height: auto; }}
  figcaption {{ margin-top: 6px; font-size: 12px; color: var(--muted); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 13px;
  }}
  th, td {{
    text-align: left;
    padding: 7px 10px;
    border-bottom: 1px solid var(--line);
    vertical-align: top;
  }}
  .kv th {{ width: 140px; color: var(--muted); font-weight: 500; }}
  .list thead th {{ color: var(--muted); font-weight: 500; background: var(--bg); }}
  .skipped ul {{ margin: 8px 0 0; padding-left: 1.2rem; color: var(--muted); font-size: 13px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>OscNet inspect · {family}</h1>
  <div class="meta">{source}</div>
</header>
<main>
  {_stage_html(payload)}
  <section class="drawers">
    {drawers_html}
    {skipped_html}
  </section>
</main>
<script id="report-data" type="application/json">{data_json}</script>
<script>
(function () {{
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const series = data.phase_series;
  const frames = (data.stage && data.stage.frames) || [];
  const nSteps = series && series.phases ? series.phases.length : (frames.length || 0);
  let timer = null;
  let t = 0;
  let selectedSite = null;
  let edgeThreshold = 0.35;

  function wrapDelta(a, b) {{
    const twoPi = Math.PI * 2;
    let d = a - b;
    d = ((d + Math.PI) % twoPi + twoPi) % twoPi - Math.PI;
    return d;
  }}

  function deltaColor(delta, limit) {{
    const lim = Math.max(limit, 1e-6);
    let x = delta / lim;
    if (x > 1) x = 1;
    if (x < -1) x = -1;
    let rr, gg, bb;
    if (x < 0) {{
      const s = x + 1;
      rr = Math.round(23 + (245 - 23) * s);
      gg = Math.round(92 + (245 - 92) * s);
      bb = Math.round(211 + (245 - 211) * s);
    }} else {{
      const s = x;
      rr = Math.round(245 + (180 - 245) * s);
      gg = Math.round(245 + (35 - 245) * s);
      bb = Math.round(245 + (24 - 245) * s);
    }}
    return "rgb(" + rr + "," + gg + "," + bb + ")";
  }}

  let deltaLimit = 1e-3;
  if (series && series.phases && series.phases.length) {{
    const base = series.phases[0];
    for (let i = 0; i < series.phases.length; i++) {{
      for (let s = 0; s < series.phases[i].length; s++) {{
        const d = Math.abs(wrapDelta(series.phases[i][s], base[s]));
        if (d > deltaLimit) deltaLimit = d;
      }}
    }}
  }}

  function sparkPoints(values, width, height, pad) {{
    if (!values || !values.length) return "";
    let lo = Math.min.apply(null, values);
    let hi = Math.max.apply(null, values);
    if (!(hi > lo)) {{ lo -= 1; hi += 1; }}
    const n = values.length;
    return values.map((v, i) => {{
      const x = pad + (n === 1 ? 0 : (i / (n - 1)) * (width - 2 * pad));
      const y = pad + (1 - ((v - lo) / (hi - lo))) * (height - 2 * pad);
      return x.toFixed(1) + "," + y.toFixed(1);
    }}).join(" ");
  }}

  function drawOrderSpark(step) {{
    const svg = document.querySelector(".order-spark");
    if (!svg || !series || !series.order_parameter) {{
      if (svg) svg.innerHTML = "";
      return;
    }}
    const values = series.order_parameter;
    const width = 120, height = 24, pad = 3;
    const markerX = pad + (values.length === 1 ? 0 : (step / Math.max(values.length - 1, 1)) * (width - 2 * pad));
    svg.innerHTML =
      '<polyline fill="none" stroke="#1c1b19" stroke-width="1.4" points="'
      + sparkPoints(values, width, height, pad) + '" />'
      + '<line x1="' + markerX.toFixed(1) + '" y1="' + pad + '" x2="'
      + markerX.toFixed(1) + '" y2="' + (height - pad)
      + '" stroke="#b42318" stroke-width="1.2" />';
  }}

  function neighborSet(site) {{
    const out = new Set([site]);
    document.querySelectorAll(".osc-edge").forEach(edge => {{
      const src = Number(edge.getAttribute("data-src"));
      const dst = Number(edge.getAttribute("data-dst"));
      if (src === site || dst === site) {{ out.add(src); out.add(dst); }}
    }});
    return out;
  }}

  function applyEdgeFilter() {{
    document.querySelectorAll(".osc-edge").forEach(edge => {{
      const strength = Number(edge.getAttribute("data-strength") || 0);
      edge.style.display = strength < edgeThreshold ? "none" : "";
    }});
  }}

  function applySelection() {{
    const neighbors = selectedSite == null ? null : neighborSet(selectedSite);
    document.querySelectorAll(".osc-node").forEach(node => {{
      const site = Number(node.getAttribute("data-site"));
      node.classList.remove("is-selected", "is-dimmed");
      if (neighbors) {{
        if (site === selectedSite) node.classList.add("is-selected");
        else if (!neighbors.has(site)) node.classList.add("is-dimmed");
      }}
    }});
    document.querySelectorAll(".osc-edge").forEach(edge => {{
      edge.classList.remove("is-hot", "is-dimmed");
      if (selectedSite == null) return;
      const src = Number(edge.getAttribute("data-src"));
      const dst = Number(edge.getAttribute("data-dst"));
      edge.classList.add((src === selectedSite || dst === selectedSite) ? "is-hot" : "is-dimmed");
    }});
    updateSiteInfo();
  }}

  function updateSiteInfo() {{
    const box = document.querySelector(".site-info");
    const spark = document.querySelector(".site-spark");
    const legend = document.querySelector(".spark-legend");
    if (!box) return;
    if (selectedSite == null) {{
      box.textContent = "Click an oscillator to inspect it.";
      if (spark) {{ spark.setAttribute("hidden", ""); spark.innerHTML = ""; }}
      if (legend) legend.setAttribute("hidden", "");
      return;
    }}
    const idx = t;
    const parts = ["site " + selectedSite];
    if (series && series.phases && series.phases[idx]) {{
      const x = series.phases[idx][selectedSite];
      const d = wrapDelta(x, series.phases[0][selectedSite]);
      parts.push((series.has_velocity ? "x=" : "phase=") + Number(x).toFixed(3));
      parts.push("Δt0=" + d.toFixed(3));
    }}
    if (series && series.has_velocity && series.velocity) {{
      parts.push("v=" + Number(series.velocity[idx][selectedSite]).toFixed(3));
    }}
    if (series && series.has_velocity && series.energy) {{
      parts.push("E=" + Number(series.energy[idx][selectedSite]).toFixed(3));
    }}
    if (series && series.omega && series.omega[selectedSite] != null) {{
      parts.push("ω=" + Number(series.omega[selectedSite]).toFixed(4));
    }}
    let degree = 0;
    document.querySelectorAll(".osc-edge").forEach(edge => {{
      const src = Number(edge.getAttribute("data-src"));
      const dst = Number(edge.getAttribute("data-dst"));
      if ((src === selectedSite || dst === selectedSite) && edge.style.display !== "none") degree += 1;
    }});
    parts.push("links=" + degree);
    box.textContent = parts.join(" · ");
    drawSiteSpark(selectedSite);
    if (legend) {{
      legend.removeAttribute("hidden");
      const x = legend.querySelector(".spark-x");
      const v = legend.querySelector(".spark-v");
      const e = legend.querySelector(".spark-e");
      if (x) x.textContent = series && series.has_velocity ? "x" : "phase";
      if (v) v.hidden = !(series && series.has_velocity);
      if (e) e.hidden = !(series && series.has_velocity);
    }}
  }}

  function drawSiteSpark(site) {{
    const svg = document.querySelector(".site-spark");
    if (!svg || !series || !series.phases) return;
    const width = 320, height = 64, pad = 6;
    const xs = series.phases.map(row => row[site]);
    const markerX = pad + (xs.length === 1 ? 0 : (t / Math.max(xs.length - 1, 1)) * (width - 2 * pad));
    svg.removeAttribute("hidden");
    let htmlSpark = '<polyline fill="none" stroke="#1c1b19" stroke-width="1.6" points="'
      + sparkPoints(xs, width, height, pad) + '" />';
    if (series.has_velocity && series.velocity) {{
      htmlSpark += '<polyline fill="none" stroke="#175cd3" stroke-width="1.3" points="'
        + sparkPoints(series.velocity.map(row => row[site]), width, height, pad) + '" />';
    }}
    if (series.has_velocity && series.energy) {{
      htmlSpark += '<polyline fill="none" stroke="#b42318" stroke-width="1.3" points="'
        + sparkPoints(series.energy.map(row => row[site]), width, height, pad) + '" />';
    }}
    htmlSpark += '<line x1="' + markerX.toFixed(1) + '" y1="' + pad + '" x2="'
      + markerX.toFixed(1) + '" y2="' + (height - pad)
      + '" stroke="#1c1b19" stroke-opacity="0.35" stroke-dasharray="2 2" />';
    svg.innerHTML = htmlSpark;
  }}

  function colorArchitecture(step) {{
    if (!series || !series.phases || !series.phases.length) return;
    const idx = Math.max(0, Math.min(series.phases.length - 1, step | 0));
    const row = series.phases[idx];
    const base = series.phases[0];
    document.querySelectorAll(".osc-node").forEach(node => {{
      const site = Number(node.getAttribute("data-site"));
      if (!Number.isFinite(site) || site < 0 || site >= row.length) return;
      const d = wrapDelta(row[site], base[site]);
      node.setAttribute("fill", deltaColor(d, deltaLimit));
      const baseR = Number(node.getAttribute("data-r") || node.getAttribute("r") || 6);
      const amp = Math.min(1, Math.abs(d) / Math.max(deltaLimit, 1e-6));
      node.setAttribute("r", String(baseR * (1 + 0.45 * amp)));
    }});
  }}

  function showFrame(step) {{
    const idx = Math.max(0, Math.min(Math.max(nSteps, 1) - 1, step | 0));
    t = idx;
    const slider = document.querySelector(".phase-scrub .slider");
    if (slider) {{
      slider.max = String(Math.max(nSteps, 1) - 1);
      slider.value = String(idx);
    }}
    const labelEl = document.querySelector(".phase-scrub .label");
    if (labelEl) labelEl.textContent = (idx + 1) + " / " + Math.max(nSteps, 1);
    const orderEl = document.querySelector(".phase-scrub .order");
    if (orderEl) {{
      orderEl.textContent = (series && series.order_parameter && series.order_parameter[idx] != null)
        ? Number(series.order_parameter[idx]).toFixed(3)
        : "—";
    }}
    drawOrderSpark(idx);
    colorArchitecture(idx);
    applySelection();
    if (frames.length) {{
      const img = document.querySelector(".frame");
      if (img) img.src = frames[Math.min(idx, frames.length - 1)].src;
    }}
    document.querySelectorAll(".multiplicity").forEach(btn => {{
      btn.classList.toggle("active", Number(btn.dataset.step) === idx);
    }});
  }}

  const slider = document.querySelector(".phase-scrub .slider");
  if (slider) slider.addEventListener("input", () => showFrame(Number(slider.value)));
  const playBtn = document.querySelector(".phase-scrub .play");
  if (playBtn) {{
    if (nSteps <= 1) {{
      playBtn.disabled = true;
      playBtn.title = "Need more than 1 settle step in the NPZ trace";
    }}
    playBtn.addEventListener("click", () => {{
      if (nSteps <= 1) return;
      if (timer) {{
        clearInterval(timer);
        timer = null;
        playBtn.textContent = "Play";
        return;
      }}
      playBtn.textContent = "Pause";
      timer = setInterval(() => {{
        let next = t + 1;
        if (next >= nSteps) next = 0;
        showFrame(next);
      }}, 350);
    }});
  }}
  const edgeSlider = document.querySelector(".edge-slider");
  if (edgeSlider) {{
    edgeSlider.addEventListener("input", () => {{
      edgeThreshold = Number(edgeSlider.value) / 100;
      applyEdgeFilter();
      applySelection();
    }});
  }}
  const clearBtn = document.querySelector(".clear-sel");
  if (clearBtn) clearBtn.addEventListener("click", () => {{
    selectedSite = null;
    applySelection();
  }});
  document.querySelectorAll(".multiplicity").forEach(btn => {{
    btn.addEventListener("click", () => showFrame(Number(btn.dataset.step)));
  }});
  document.addEventListener("click", (ev) => {{
    const node = ev.target.closest && ev.target.closest(".osc-node");
    if (!node) return;
    const site = Number(node.getAttribute("data-site"));
    if (!Number.isFinite(site)) return;
    selectedSite = (selectedSite === site) ? null : site;
    applySelection();
  }});

  applyEdgeFilter();
  if (nSteps > 0) showFrame(0);
  else applySelection();
}})();
</script>
</body>
</html>
"""
