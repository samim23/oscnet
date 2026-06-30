from pathlib import Path

from oscnet.analysis.generator_frontier import (
    infer_generator_variant,
    infer_generator_seed,
    paired_generator_metric_deltas,
    read_generator_sweep_csv,
    summarize_generator_frontier,
    write_frontier_csv,
    write_frontier_markdown,
    write_paired_delta_csv,
    write_paired_delta_markdown,
)


def test_infer_generator_variant_handles_known_modal_sweeps():
    assert (
        infer_generator_variant(
            "mnist_generator_sparse_horn_state_mlp_strength8_diversity_"
            "horn_recommended_n196_resizeconv_train500_seed11_20e"
        )
        == "horn_recommended"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_sparse_horn_recommended_ablation_"
            "no_main_coupling_n196_resizeconv_train500_seed11_20e"
        )
        == "no_main_coupling"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_fashion_mnist_frontier_"
            "state_mlp_strength8_dist005_n196_resizeconv_train500_seed11_20e"
        )
        == "state_mlp_strength8_dist005"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_fashion_mnist_readout_capacity_"
            "horn_ch16_n196_resizeconv16_train500_seed11_20e"
        )
        == "horn_ch16"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_fashion_mnist_horn_calibration_"
            "horn_dist0025_n196_resizeconv_train500_seed11_20e"
        )
        == "horn_dist0025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_gray_frontier_"
            "horn_recommended_n256_resizeconv_train1000_seed11_20e"
        )
        == "horn_recommended"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_gray_convjudge_frontier_"
            "state_mlp_strength8_n256_resizeconv_train1000_seed11_20e"
        )
        == "state_mlp_strength8"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_frontier_"
            "horn_dist005_n256_resizeconv_train1000_seed11_20e"
        )
        == "horn_dist005"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_feature_metrics_"
            "state_mlp_strength8_n256_resizeconv_train1000_seed11_20e"
        )
        == "state_mlp_strength8"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_attribution_"
            "horn_no_main_interaction_n256_resizeconv_train1000_seed11_20e"
        )
        == "horn_no_main_interaction"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_sparse_drive_"
            "horn_no_main_drive025_n256_resizeconv_train1000_seed11_20e"
        )
        == "horn_no_main_drive025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_sparse_drive_seed_repeat_"
            "horn_drive025_n256_resizeconv_train1000_seed37_20e"
        )
        == "horn_drive025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_judge_audit_"
            "horn_prefix025_resconvjudge_n256_resizeconv_train1000_seed11_20e"
        )
        == "horn_prefix025_resconvjudge"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_semantic_feature_drift_"
            "horn_resfeat025_n256_resizeconv_train2000_seed11_20e"
        )
        == "horn_resfeat025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_semantic_feature_drift_attribution_"
            "horn_no_main_resfeat025_n256_resizeconv_train2000_seed23_20e"
        )
        == "horn_no_main_resfeat025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_attractor_robustness_"
            "state_mlp_resfeat025_n256_resizeconv_train2000_seed11_20e"
        )
        == "state_mlp_resfeat025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat_"
            "horn_resfeat025_n256_resizeconv_train2000_seed23_20e"
        )
        == "horn_resfeat025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_main_coupling_strength_"
            "horn_resfeat025_main025_n256_resizeconv_train2000_seed11_20e"
        )
        == "horn_resfeat025_main025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat_"
            "state_mlp_resfeat025_n256_resizeconv_train2000_seed23_20e"
        )
        == "state_mlp_resfeat025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_main_coupling_fine_"
            "horn_resfeat025_main075_n256_resizeconv_train2000_seed23_20e"
        )
        == "horn_resfeat025_main075"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_main_coupling_current_"
            "horn_resfeat025_main050_n256_resizeconv_train2000_seed23_20e"
        )
        == "horn_resfeat025_main050"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_normdist_"
            "horn_resfeat025_main025_n256_resizeconv_train2000_seed23_20e"
        )
        == "horn_resfeat025_main025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_normlocal_"
            "horn_resfeat025_main050_n256_resizeconv_train2000_seed23_20e"
        )
        == "horn_resfeat025_main050"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_normlocal_radius_"
            "horn_resfeat025_r024_n256_resizeconv_train2000_seed23_20e"
        )
        == "horn_resfeat025_r024"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_coarse_to_fine_"
            "coarse16_c2f100_n256_resizeconv_train2000_seed23_20e"
        )
        == "coarse16_c2f100"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_"
            "coarse16_c2f025_dist050_n256_resizeconv_train2000_seed11_20e"
        )
        == "coarse16_c2f025_dist050"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat_"
            "coarse16_c2f025_local050_n256_resizeconv_train2000_seed41_20e"
        )
        == "coarse16_c2f025_local050"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_coarse_to_fine_conversion_"
            "coarse16_c2f025_local050_ch32_dist0025_"
            "n256_resizeconv_train2000_seed23_20e"
        )
        == "coarse16_c2f025_local050_ch32_dist0025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_coarse_to_fine_feedback_"
            "coarse16_c2f025_local050_ch32_dist0025_feedback050_"
            "n256_resizeconv_train2000_seed23_20e"
        )
        == "coarse16_c2f025_local050_ch32_dist0025_feedback050"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_structured_drive_"
            "horn_grid025_n256_resizeconv_train1000_seed23_20e"
        )
        == "horn_grid025"
    )
    assert (
        infer_generator_variant(
            "mnist_generator_cifar10_rgb_coherent_drive_"
            "horn_center025_n256_resizeconv_train1000_seed23_20e"
        )
        == "horn_center025"
    )
    assert (
        infer_generator_seed(
            "mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat_"
            "coarse16_c2f025_local050_n256_resizeconv_train2000_seed41_20e"
        )
        == 41
    )
    assert infer_generator_seed("run_without_seed") is None


def test_generator_frontier_marks_non_dominated_tradeoff(tmp_path: Path):
    rows = {
        "horn": [
            {
                "generator.classifier_label_accuracy": "1.0",
                "generator.diversity_ratio": "1.10",
                "generator.nearest_real_mse": "0.052",
                "generator.attractor_robustness.label_accuracy": "0.75",
                "generator.attractor_robustness.pixel_within_class_pairwise_mse": "4.0",
                "generator.success_diagnostics.coarse_state_energy_final": "0.3",
                "generator.success_diagnostics.coarse_state_update_rms_settling_ratio": "0.5",
                "generator.success_diagnostics.coarse_coupling_potential_proxy_delta": "-0.1",
                "generator.success_diagnostics.coarse_to_fine_potential_proxy_delta": "-0.2",
                "generator.success_diagnostics.coarse_to_fine_profile_density": "0.25",
            },
            {
                "generator.classifier_label_accuracy": "1.0",
                "generator.diversity_ratio": "1.20",
                "generator.nearest_real_mse": "0.056",
                "generator.attractor_robustness.label_accuracy": "0.80",
                "generator.attractor_robustness.pixel_within_class_pairwise_mse": "5.0",
            },
        ],
        "state_mlp": [
            {
                "generator.classifier_label_accuracy": "1.0",
                "generator.diversity_ratio": "0.76",
                "generator.nearest_real_mse": "0.034",
            }
        ],
        "dominated": [
            {
                "generator.classifier_label_accuracy": "0.9",
                "generator.diversity_ratio": "0.5",
                "generator.nearest_real_mse": "0.080",
            }
        ],
    }

    summaries = summarize_generator_frontier(rows, accuracy_floor=0.99)
    by_variant = {summary.variant: summary for summary in summaries}

    assert by_variant["horn"].pareto_frontier is True
    assert by_variant["state_mlp"].pareto_frontier is True
    assert by_variant["dominated"].pareto_frontier is False
    assert by_variant["horn"].attractor_pixel_diversity_score_mean > 0.0
    assert by_variant["horn"].coarse_state_energy_mean == 0.3
    assert by_variant["horn"].coarse_to_fine_potential_delta_mean == -0.2

    csv_path = tmp_path / "frontier.csv"
    md_path = tmp_path / "frontier.md"
    write_frontier_csv(summaries, csv_path)
    write_frontier_markdown(summaries, md_path)

    assert "pareto_frontier" in csv_path.read_text()
    markdown = md_path.read_text()
    assert "Update settle" in markdown
    assert "Basin score" in markdown
    assert "C2F delta" in markdown
    assert "Frontier variants" in markdown


def test_read_generator_sweep_csv_groups_by_variant(tmp_path: Path):
    csv_path = tmp_path / "sweep.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run,generator.classifier_label_accuracy,"
                "generator.diversity_ratio,generator.nearest_real_mse",
                "prefix_state_mlp_strength8_diversity_horn_n196_seed11,"
                "1.0,1.1,0.05",
                "prefix_state_mlp_strength8_diversity_horn_n196_seed12,"
                "1.0,1.2,0.06",
            ]
        )
        + "\n"
    )

    grouped = read_generator_sweep_csv(csv_path)

    assert list(grouped) == ["horn"]
    assert len(grouped["horn"]) == 2


def test_paired_generator_metric_deltas_compare_matched_seeds(tmp_path: Path):
    csv_path = tmp_path / "paired.csv"
    csv_path.write_text(
        "\n".join(
            [
                "run,generator.classifier_label_accuracy,"
                "generator.diversity_ratio,generator.nearest_real_mse",
                "prefix_cifar10_rgb_coarse_to_fine_local_repeat_"
                "coarse16_c2f000_n256_seed11,0.40,0.80,0.030",
                "prefix_cifar10_rgb_coarse_to_fine_local_repeat_"
                "coarse16_c2f000_n256_seed23,0.50,0.90,0.020",
                "prefix_cifar10_rgb_coarse_to_fine_local_repeat_"
                "coarse16_c2f025_local050_n256_seed11,0.45,0.85,0.029",
                "prefix_cifar10_rgb_coarse_to_fine_local_repeat_"
                "coarse16_c2f025_local050_n256_seed23,0.40,0.95,0.025",
                "prefix_cifar10_rgb_coarse_to_fine_local_repeat_"
                "horn_normlocal_n256_seed11,0.30,1.00,0.035",
            ]
        )
        + "\n"
    )

    grouped = read_generator_sweep_csv(csv_path)
    summaries = paired_generator_metric_deltas(
        grouped,
        baseline_variant="coarse16_c2f000",
        target_variants=["coarse16_c2f025_local050", "horn_normlocal"],
    )
    by_target_metric = {
        (summary.target_variant, summary.metric): summary
        for summary in summaries
    }

    acc = by_target_metric[
        ("coarse16_c2f025_local050", "generator.classifier_label_accuracy")
    ]
    assert acc.paired_runs == 2
    assert acc.seeds == "11,23"
    assert round(acc.delta_mean, 4) == -0.025
    assert acc.target_wins == 1

    mse = by_target_metric[
        ("coarse16_c2f025_local050", "generator.nearest_real_mse")
    ]
    assert mse.direction == "lower"
    assert round(mse.delta_mean, 4) == 0.002
    assert mse.target_wins == 1

    horn_acc = by_target_metric[
        ("horn_normlocal", "generator.classifier_label_accuracy")
    ]
    assert horn_acc.paired_runs == 1
    assert horn_acc.seeds == "11"

    csv_out = tmp_path / "paired_deltas.csv"
    md_out = tmp_path / "paired_deltas.md"
    write_paired_delta_csv(summaries, csv_out)
    write_paired_delta_markdown(summaries, md_out)

    assert "delta_mean" in csv_out.read_text()
    markdown = md_out.read_text()
    assert "target - baseline" in markdown
    assert "coarse16_c2f025_local050" in markdown
