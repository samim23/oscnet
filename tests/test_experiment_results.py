import csv
import json

from oscnet.experiments.results import (
    collect_experiment_summaries,
    find_experiment_runs,
    format_comparison_table,
    load_experiment_summary,
    write_comparison_csv,
)


def _write_summary(root, name, summary):
    run = root / name
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    with open(metrics / "summary.json", "w") as f:
        json.dump(summary, f)
    return run


def test_experiment_result_comparison_loads_sorts_and_writes(tmp_path):
    slow = _write_summary(
        tmp_path,
        "slow_run",
        {
            "final_eval_loss": 0.2,
            "best_loss": 0.18,
            "best_epoch": 2,
            "quality": {"pixel_correlation": 0.7, "foreground_f1": 0.6},
        },
    )
    fast = _write_summary(
        tmp_path,
        "fast_run",
        {
            "final_eval_loss": 0.1,
            "best_loss": 0.08,
            "best_epoch": 3,
            "quality": {"pixel_correlation": 0.9, "foreground_f1": 0.8},
        },
    )
    missing = _write_summary(
        tmp_path,
        "missing_eval",
        {"quality": {"pixel_correlation": 0.4}},
    )

    runs = find_experiment_runs(tmp_path)
    assert runs == [fast, missing, slow]

    row = load_experiment_summary(
        fast / "metrics" / "summary.json",
        metric_names=("final_eval_loss", "quality.pixel_correlation"),
    )
    assert row.run == "fast_run"
    assert row.metrics["final_eval_loss"] == 0.1
    assert row.metrics["quality.pixel_correlation"] == 0.9

    rows = collect_experiment_summaries(
        [slow, missing, fast],
        metric_names=("final_eval_loss", "quality.pixel_correlation"),
        sort_by="final_eval_loss",
    )
    assert [row.run for row in rows] == ["fast_run", "slow_run", "missing_eval"]

    rows_desc = collect_experiment_summaries(
        [slow, missing, fast],
        metric_names=("final_eval_loss", "quality.pixel_correlation"),
        sort_by="quality.pixel_correlation",
        descending=True,
    )
    assert [row.run for row in rows_desc] == ["fast_run", "slow_run", "missing_eval"]

    table = format_comparison_table(
        rows,
        metric_names=("final_eval_loss", "quality.pixel_correlation"),
    )
    assert "| fast_run | 0.1 | 0.9 |" in table

    csv_path = tmp_path / "comparison.csv"
    write_comparison_csv(
        rows,
        csv_path,
        metric_names=("final_eval_loss", "quality.pixel_correlation"),
    )
    with open(csv_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))

    assert csv_rows[0]["run"] == "fast_run"
    assert csv_rows[0]["final_eval_loss"] == "0.1"
