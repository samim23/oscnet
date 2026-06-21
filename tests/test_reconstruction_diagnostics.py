import json

import numpy as np

from oscnet.analysis import (
    infer_changed_mask,
    summarize_reconstruction_artifact,
    summarize_run_diagnostics,
)


def test_reconstruction_artifact_summary_separates_changed_region(tmp_path):
    artifact_path = tmp_path / "mnist_reconstructions_epoch_001.npz"
    inputs = np.array([[[0.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    originals = np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    reconstructions = np.array([[[0.75, 1.0], [0.2, 0.0]]], dtype=np.float32)
    np.savez(
        artifact_path,
        inputs=inputs,
        originals=originals,
        reconstructions=reconstructions,
    )

    changed = infer_changed_mask(inputs, originals)
    summary = summarize_reconstruction_artifact(artifact_path, run="tiny")

    assert changed.tolist() == [[[True, False], [False, False]]]
    assert summary.run == "tiny"
    assert summary.n_examples == 1
    assert summary.changed_fraction == 0.25
    assert np.isclose(summary.changed_input_mse, 1.0)
    assert np.isclose(summary.changed_mse, 0.0625)
    assert np.isclose(summary.changed_improvement, 0.9375)
    assert np.isclose(summary.unchanged_mse, (0.0 + 0.04 + 0.0) / 3.0)


def test_run_diagnostics_include_history_and_summary(tmp_path):
    run_root = tmp_path / "run"
    artifact_dir = run_root / "artifacts"
    metrics_dir = run_root / "metrics"
    artifact_dir.mkdir(parents=True)
    metrics_dir.mkdir()
    np.savez(
        artifact_dir / "mnist_reconstructions_epoch_002.npz",
        inputs=np.zeros((1, 2, 2), dtype=np.float32),
        originals=np.ones((1, 2, 2), dtype=np.float32),
        reconstructions=np.full((1, 2, 2), 0.5, dtype=np.float32),
    )
    with open(metrics_dir / "summary.json", "w") as f:
        json.dump({"final_eval_loss": 0.2, "best_loss": 0.1}, f)
    with open(metrics_dir / "history.json", "w") as f:
        json.dump({"grad_norm": [0.5, 2.0, 1.5]}, f)

    summary = summarize_run_diagnostics(run_root)

    assert summary.run == "run"
    assert summary.final_eval_loss == 0.2
    assert summary.best_loss == 0.1
    assert summary.max_grad_norm == 2.0
    assert summary.final_grad_norm == 1.5
    assert np.isclose(summary.changed_mse, 0.25)
