"""Tests for oscnet.inspection adapters and report composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscnet.inspection import (
    detect_family,
    inspect_trace,
    load_trace,
)
from oscnet.inspection.adapters import load_npz_arrays
from oscnet.inspection.geometry import infer_grid_shape, trajectory_keyframes
from oscnet.inspection.views import get_views


def _write_generator_npz(path: Path, *, with_velocity: bool = False) -> Path:
    n_steps, batch, n_sites = 4, 3, 16
    payload = {
        "initial_theta": np.zeros((batch, n_sites), dtype=np.float32),
        "theta_trajectory": np.linspace(0, 1, n_steps * batch * n_sites, dtype=np.float32).reshape(
            n_steps, batch, n_sites
        ),
        "final_theta": np.ones((batch, n_sites), dtype=np.float32),
        "generated": np.random.default_rng(0).random((batch, 64), dtype=np.float32),
        "omega": np.linspace(0.5, 1.5, n_sites, dtype=np.float32),
        "coupling": np.eye(n_sites, dtype=np.float32) * 0.2,
        "coupling_profile": np.ones((n_sites, n_sites), dtype=np.float32),
    }
    if with_velocity:
        payload["velocity_trajectory"] = np.zeros((n_steps, batch, n_sites), dtype=np.float32)
    np.savez(path, **payload)
    return path


def _write_winfree_npz(path: Path) -> Path:
    n_steps, batch, sites, channels = 3, 2, 16, 4
    np.savez(
        path,
        decoder_thetas=np.zeros((n_steps, batch, sites, channels), dtype=np.float32),
        decoder_final_theta=np.zeros((batch, sites, channels), dtype=np.float32),
        decoder_energies=np.linspace(1, 0.2, n_steps * batch, dtype=np.float32).reshape(
            n_steps, batch
        ),
        decoder_omega=np.ones((batch, sites, channels), dtype=np.float32),
        latent=np.zeros((batch, 4), dtype=np.float32),
    )
    return path


def _write_phase_flow_npz(path: Path) -> Path:
    n_steps, batch, height, width, channels = 2, 2, 8, 8, 2
    np.savez(
        path,
        initial_theta=np.zeros((batch, height, width, channels), dtype=np.float32),
        theta_trajectory=np.zeros(
            (n_steps, batch, height, width, channels), dtype=np.float32
        ),
        rate_trajectory=np.ones(
            (n_steps, batch, height, width, channels), dtype=np.float32
        ),
        final_theta=np.zeros((batch, height, width, channels), dtype=np.float32),
        predicted_clean=np.random.default_rng(1).random((batch, 64), dtype=np.float32),
    )
    return path


def test_infer_grid_shape_square_and_hint():
    assert infer_grid_shape(16) == (4, 4)
    assert infer_grid_shape(12) is None
    assert infer_grid_shape(12, hint=(3, 4)) == (3, 4)
    with pytest.raises(ValueError):
        infer_grid_shape(12, hint=(2, 2))


def test_oscillator_pack_matches_model_coupling_grid():
    from oscnet.core.coupling import oscillator_grid_coordinates
    from oscnet.inspection.geometry import (
        oscillator_pack_shape,
        oscillator_site_positions,
    )

    assert oscillator_pack_shape(98) == (9, 11)
    positions = oscillator_site_positions(98)
    assert len(positions) == 98
    model = np.asarray(oscillator_grid_coordinates(98))
    # model columns are (y, x); inspector stores (x, y)
    got = np.asarray([[y, x] for x, y in positions], dtype=np.float64)
    np.testing.assert_allclose(got, model, atol=1e-5)


def test_architecture_uses_model_pack_not_fake_rectangle():
    """98 sites must not be forced into a filled 7×14 schematic."""

    from oscnet.inspection.views.architecture import render_architecture_svg
    from oscnet.inspection import load_trace

    path = Path(
        "outputs/smoke/fashion_mnist_generator_horn/traces/mnist_generator_trace_epoch_001.npz"
    )
    if not path.is_file():
        pytest.skip("missing horn smoke trace")
    svg = render_architecture_svg(load_trace(path))
    assert "9×11 pack · 98 sites" in svg
    assert "7×14" not in svg


def test_trajectory_keyframes_includes_ends():
    assert trajectory_keyframes(1) == (0,)
    assert trajectory_keyframes(5, max_frames=5) == (0, 1, 2, 3, 4)
    keys = trajectory_keyframes(10, max_frames=4)
    assert keys[0] == 0
    assert keys[-1] == 9
    assert len(keys) == 4


def test_detect_and_load_generator(tmp_path):
    path = _write_generator_npz(tmp_path / "gen.npz", with_velocity=True)
    arrays = load_npz_arrays(path)
    assert detect_family(arrays) == "generator"
    bundle = load_trace(path)
    assert bundle.family == "generator"
    assert bundle.phase_trajectory.shape == (4, 3, 16)
    assert bundle.velocity_trajectory is not None
    assert bundle.coupling.shape == (16, 16)
    assert bundle.grid_shape == (4, 4)
    assert bundle.readout.shape == (3, 64)


def test_detect_and_load_winfree(tmp_path):
    path = _write_winfree_npz(tmp_path / "win.npz")
    bundle = load_trace(path, grid_shape=(4, 4))
    assert bundle.family == "winfree"
    assert bundle.meta["prefix"] == "decoder"
    assert bundle.phase_trajectory.shape == (3, 2, 16, 4)
    assert bundle.energies.shape == (3, 2)
    assert bundle.grid_shape == (4, 4)


def test_detect_and_load_phase_flow(tmp_path):
    path = _write_phase_flow_npz(tmp_path / "flow.npz")
    bundle = load_trace(path)
    assert bundle.family == "phase_flow"
    assert bundle.grid_shape == (8, 8)
    assert bundle.rate_trajectory is not None


def test_inspect_trace_writes_expected_artifacts(tmp_path):
    path = _write_generator_npz(tmp_path / "gen.npz")
    out = tmp_path / "report"
    report = inspect_trace(path, out, batch_index=1, max_frames=4)
    assert report.family == "generator"
    assert (out / "manifest.json").is_file()
    assert (out / "overview.json").is_file()
    assert (out / "overview.txt").is_file()
    assert not (out / "overview.png").exists()
    assert (out / "architecture.svg").is_file()
    assert "<svg" in (out / "architecture.svg").read_text()
    assert (out / "coupling.png").is_file()
    assert (out / "phase_fields.png").is_file()
    assert (out / "synchrony.png").is_file()
    assert (out / "omega.png").is_file()
    assert (out / "readout.png").is_file()
    assert report.html_path is not None
    assert report.html_path.is_file()
    html = report.html_path.read_text()
    assert "oscillator field" in html
    assert "live-arch" in html
    assert "osc-node" in html
    assert "osc-edge" in (out / "architecture.svg").read_text()
    assert "phase_series" in html
    assert "order_parameter" in html
    assert "class=\"stage\"" in html
    assert "arch-band" in html
    assert "class=\"transport" in html
    assert "Phase field movie" in html
    assert 'id="drawer-fields"' in html
    assert "Short trace" not in html
    assert "readout-band" not in html
    assert "Readout (decoder output)" in html
    assert "phase-scrub" in html
    assert "multiplicity" in html
    assert "site-info" in html
    assert "order-spark" in html
    assert "max-width: none" in html
    assert "Trace keys" in html
    assert "Coupling matrix" in html
    assert 'role="tablist"' not in html
    assert "data-tab=" not in html
    assert (out / "architecture.svg").read_text().count("osc-node") >= 1
    assert "single-page report" not in html
    assert "border-radius: 999px" not in html
    skipped_names = {item["view"] for item in report.skipped}
    assert "rate_fields" in skipped_names
    assert "vertical_gain" in skipped_names


def test_inspect_trace_can_skip_html(tmp_path):
    path = _write_generator_npz(tmp_path / "gen.npz")
    out = tmp_path / "no_html"
    report = inspect_trace(path, out, html=False)
    assert report.html_path is None
    assert not (out / "index.html").exists()


def test_site_series_includes_horn_velocity_energy(tmp_path):
    from oscnet.inspection.views.architecture import site_phase_series
    from oscnet.inspection import load_trace

    path = _write_generator_npz(tmp_path / "horn.npz", with_velocity=True)
    bundle = load_trace(path)
    series = site_phase_series(bundle, batch_index=0, max_frames=4)
    assert series is not None
    assert series["has_velocity"] is True
    assert "velocity" in series and "energy" in series
    assert np.shape(series["velocity"]) == np.shape(series["phases"])
    assert np.shape(series["energy"]) == np.shape(series["phases"])
    html = inspect_trace(path, tmp_path / "horn_report").html_path.read_text()
    assert "site-spark" in html
    assert "has_velocity" in html


def test_inspect_winfree_energy_panel(tmp_path):
    path = _write_winfree_npz(tmp_path / "win.npz")
    out = tmp_path / "win_report"
    report = inspect_trace(path, out, grid_shape=(4, 4), views=["overview", "synchrony"])
    assert (out / "synchrony.png").is_file()
    assert (out / "energy.png").is_file()
    assert report.family == "winfree"


def test_architecture_caption_marks_spatial_coupling(tmp_path):
    from oscnet.inspection.views.architecture import render_architecture_svg
    from oscnet.inspection import load_trace

    path = _write_phase_flow_npz(tmp_path / "flow.npz")
    svg = render_architecture_svg(load_trace(path))
    assert "spatial / implicit" in svg
    assert '<line class="osc-edge' not in svg


def test_architecture_draws_edges_for_large_local_coupling(tmp_path):
    """Large banks used to skip all edges (N>128) while still claiming links."""

    from oscnet.inspection.views.architecture import render_architecture_svg
    from oscnet.inspection import load_trace

    path = Path(
        "outputs/smoke/recovery_mixed_smoke/traces/mnist_generator_trace_epoch_001.npz"
    )
    if not path.is_file():
        pytest.skip("missing recovery smoke trace")
    svg = render_architecture_svg(load_trace(path))
    assert svg.count("osc-edge") >= 20
    assert "links" in svg
    assert "coupling not drawn" not in svg


def test_get_views_rejects_unknown():
    with pytest.raises(ValueError):
        get_views(["overview", "not_a_view"])


@pytest.mark.parametrize(
    "relative",
    [
        "outputs/reference/mnist_generator_kuramoto_n32_seed11_2e_smoke/traces/mnist_generator_trace_epoch_002.npz",
        "outputs/smoke/fashion_mnist_generator_horn/traces/mnist_generator_trace_epoch_001.npz",
        "outputs/reference/mnist_winfree_field_smoke/traces/mnist_latent_state_epoch_001.npz",
        "outputs/smoke/mnist_phase_flow_signed_distance_flow/traces/mnist_phase_flow_trace_epoch_001.npz",
    ],
)
def test_inspect_real_reference_traces_if_present(tmp_path, relative):
    path = Path(relative)
    if not path.is_file():
        pytest.skip(f"missing reference trace {relative}")
    kwargs = {}
    if "winfree" in relative:
        kwargs["grid_shape"] = (4, 4)
    report = inspect_trace(path, tmp_path / "real", **kwargs)
    assert report.family in {"generator", "winfree", "phase_flow"}
    assert (tmp_path / "real" / "manifest.json").is_file()
    assert len(report.artifacts) >= 2
