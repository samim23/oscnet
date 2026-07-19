"""Phase / rate / gain field movies as keyframe strips."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..geometry import (
    as_spatial_field,
    reduce_channels,
    select_batch,
    trajectory_keyframes,
)
from ..schema import TraceBundle
from ._plotting import imshow_phase, imshow_signed, save_figure
from .base import ViewContext, ViewResult

Array = np.ndarray


class PhaseFieldView:
    """Keyframe strip of phase over settle steps for one batch item."""

    name = "phase_fields"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        if bundle.phase_trajectory is None:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="no phase_trajectory",
            )
        frames, titles = extract_phase_frames(
            bundle.phase_trajectory,
            batch_index=context.batch_index,
            grid_shape=bundle.grid_shape,
            max_frames=context.max_frames,
        )
        if not frames:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="phase_trajectory produced no frames",
            )
        path = _save_keyframe_strip(
            frames,
            titles,
            context.output_dir / "phase_fields.png",
            title=f"Phase fields · {bundle.family} · batch={context.batch_index}",
            mode="phase",
            dpi=context.dpi,
        )
        return ViewResult(name=self.name, paths=[path])


class RateFieldView:
    name = "rate_fields"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        if bundle.rate_trajectory is None:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="no rate_trajectory",
            )
        frames, titles = extract_spatial_frames(
            bundle.rate_trajectory,
            batch_index=context.batch_index,
            grid_shape=bundle.grid_shape,
            max_frames=context.max_frames,
            already_spatial=bundle.family == "phase_flow",
        )
        if not frames:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="rate_trajectory produced no frames",
            )
        path = _save_keyframe_strip(
            frames,
            titles,
            context.output_dir / "rate_fields.png",
            title=f"Rate fields · {bundle.family} · batch={context.batch_index}",
            mode="signed",
            dpi=context.dpi,
        )
        return ViewResult(name=self.name, paths=[path])


class VerticalGainView:
    name = "vertical_gain"

    def render(self, bundle: TraceBundle, context: ViewContext) -> ViewResult:
        if bundle.vertical_gain_trajectory is None:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="no vertical_gain_trajectory",
            )
        frames, titles = extract_spatial_frames(
            bundle.vertical_gain_trajectory,
            batch_index=context.batch_index,
            grid_shape=bundle.grid_shape,
            max_frames=context.max_frames,
            already_spatial=False,
        )
        if not frames:
            return ViewResult(
                name=self.name,
                skipped=True,
                reason="vertical_gain_trajectory produced no frames",
            )
        path = _save_keyframe_strip(
            frames,
            titles,
            context.output_dir / "vertical_gain.png",
            title=f"Vertical gain · {bundle.family} · batch={context.batch_index}",
            mode="signed",
            dpi=context.dpi,
        )
        return ViewResult(name=self.name, paths=[path])


def extract_phase_frames(
    trajectory: Array,
    *,
    batch_index: int,
    grid_shape,
    max_frames: int,
) -> Tuple[list, list]:
    """Return ``(frames, titles)`` for a phase trajectory."""

    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 5:
        # (T, B, H, W, C)
        selected = select_batch(trajectory, batch_index, batch_axis=1)
        indices = trajectory_keyframes(selected.shape[0], max_frames=max_frames)
        frames = []
        titles = []
        for index in indices:
            frame = reduce_channels(selected[index], channel_axis=-1)
            frames.append(np.asarray(frame))
            titles.append(f"t={index}")
        return frames, titles

    if trajectory.ndim == 4:
        # (T, B, sites, channels) Winfree
        selected = select_batch(trajectory, batch_index, batch_axis=1)
        indices = trajectory_keyframes(selected.shape[0], max_frames=max_frames)
        frames = []
        titles = []
        for index in indices:
            site_field = reduce_channels(selected[index], channel_axis=-1)
            spatial, _ = as_spatial_field(site_field, grid_shape=grid_shape, site_axis=0)
            if spatial.ndim == 1:
                spatial = spatial.reshape(1, -1)
            frames.append(spatial)
            titles.append(f"t={index}")
        return frames, titles

    if trajectory.ndim == 3:
        # (T, B, sites) generator
        selected = select_batch(trajectory, batch_index, batch_axis=1)
        indices = trajectory_keyframes(selected.shape[0], max_frames=max_frames)
        frames = []
        titles = []
        for index in indices:
            spatial, _ = as_spatial_field(selected[index], grid_shape=grid_shape, site_axis=0)
            if spatial.ndim == 1:
                spatial = spatial.reshape(1, -1)
            frames.append(spatial)
            titles.append(f"t={index}")
        return frames, titles

    return [], []


def extract_spatial_frames(
    trajectory: Array,
    *,
    batch_index: int,
    grid_shape,
    max_frames: int,
    already_spatial: bool,
) -> Tuple[list, list]:
    trajectory = np.asarray(trajectory)
    if already_spatial or trajectory.ndim == 5:
        return extract_phase_frames(
            trajectory,
            batch_index=batch_index,
            grid_shape=grid_shape,
            max_frames=max_frames,
        )
    if trajectory.ndim >= 3:
        return extract_phase_frames(
            trajectory if trajectory.ndim == 3 else trajectory,
            batch_index=batch_index,
            grid_shape=grid_shape,
            max_frames=max_frames,
        )
    return [], []


def _save_keyframe_strip(
    frames,
    titles,
    path,
    *,
    title: str,
    mode: str,
    dpi: int,
):
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6), squeeze=False)
    for ax, frame, frame_title in zip(axes[0], frames, titles):
        if mode == "phase":
            imshow_phase(ax, frame, title=frame_title)
        else:
            imshow_signed(ax, frame, title=frame_title)
    fig.suptitle(title)
    return save_figure(fig, path, dpi=dpi)
