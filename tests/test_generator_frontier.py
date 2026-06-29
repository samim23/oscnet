from pathlib import Path

from oscnet.analysis.generator_frontier import (
    infer_generator_variant,
    read_generator_sweep_csv,
    summarize_generator_frontier,
    write_frontier_csv,
    write_frontier_markdown,
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


def test_generator_frontier_marks_non_dominated_tradeoff(tmp_path: Path):
    rows = {
        "horn": [
            {
                "generator.classifier_label_accuracy": "1.0",
                "generator.diversity_ratio": "1.10",
                "generator.nearest_real_mse": "0.052",
                "generator.attractor_robustness.label_accuracy": "0.75",
                "generator.attractor_robustness.pixel_within_class_pairwise_mse": "4.0",
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

    csv_path = tmp_path / "frontier.csv"
    md_path = tmp_path / "frontier.md"
    write_frontier_csv(summaries, csv_path)
    write_frontier_markdown(summaries, md_path)

    assert "pareto_frontier" in csv_path.read_text()
    markdown = md_path.read_text()
    assert "Update settle" in markdown
    assert "Basin score" in markdown
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
