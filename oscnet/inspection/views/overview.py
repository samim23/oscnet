"""Overview panel: family, slots, shapes."""

from __future__ import annotations

import json

from ..schema import TraceBundle
from .base import ViewContext, ViewResult


class OverviewView:
    name = "overview"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        overview = bundle.overview()
        json_path = context.output_dir / "overview.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(overview, indent=2, sort_keys=True) + "\n")

        lines = [
            f"family: {bundle.family}",
            f"source: {bundle.source}",
            f"grid_shape: {bundle.grid_shape}",
            "",
            "normalized slots:",
        ]
        for ref in bundle.slots():
            lines.append(f"  - {ref.name}: {tuple(ref.data.shape)} ({ref.kind})")
        lines.append("")
        lines.append(f"raw keys ({len(bundle.keys)}):")
        for key in bundle.keys:
            shape = tuple(bundle.arrays[key].shape)
            lines.append(f"  - {key}: {shape}")

        text_path = context.output_dir / "overview.txt"
        text_path.write_text("\n".join(lines) + "\n")
        return ViewResult(name=self.name, paths=[json_path, text_path])