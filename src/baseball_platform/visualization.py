"""모델 비교 결과를 재현 가능한 정적 차트로 저장한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from baseball_platform.evaluation import FoldResult, LeaderboardRow


COLORS = {
    "mean_probability": "#94A3B8",
    "logistic_regression": "#2563EB",
}


def create_model_comparison_dashboard(
    output_path: str | Path,
    fold_results: list[FoldResult],
    leaderboard: list[LeaderboardRow],
) -> Path:
    """평균 성능과 시즌별 안정성을 한 이미지에 표시한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    figure.suptitle(
        "Control Success Probability - Model Comparison",
        fontsize=16,
        fontweight="bold",
    )

    models = [row.model for row in leaderboard]
    labels = [_display_name(model) for model in models]
    colors = [COLORS.get(model, "#0F766E") for model in models]

    axes[0].bar(
        labels,
        [row.mean_log_loss for row in leaderboard],
        color=colors,
        yerr=[row.std_log_loss for row in leaderboard],
        capsize=5,
    )
    axes[0].set_title("Mean Log Loss (lower is better)")
    axes[0].set_ylabel("Log Loss")
    axes[0].grid(axis="y", alpha=0.25)
    _annotate_bars(axes[0])

    for model in models:
        model_results = sorted(
            (
                result
                for result in fold_results
                if result.model == model
            ),
            key=lambda result: result.validation_season,
        )
        axes[1].plot(
            [result.validation_season for result in model_results],
            [result.log_loss for result in model_results],
            marker="o",
            linewidth=2,
            label=_display_name(model),
            color=COLORS.get(model, "#0F766E"),
        )
    axes[1].set_title("Log Loss by Validation Season")
    axes[1].set_xlabel("Validation season")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_xticks(
        sorted({result.validation_season for result in fold_results})
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    x_positions = range(len(models))
    width = 0.36
    axes[2].bar(
        [position - width / 2 for position in x_positions],
        [row.mean_brier_score for row in leaderboard],
        width,
        label="Brier Score",
        color="#F59E0B",
    )
    axes[2].bar(
        [position + width / 2 for position in x_positions],
        [row.mean_roc_auc for row in leaderboard],
        width,
        label="ROC-AUC",
        color="#10B981",
    )
    axes[2].set_title("Supporting Metrics")
    axes[2].set_xticks(list(x_positions), labels)
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(frameon=False)

    figure.text(
        0.5,
        0.01,
        "Synthetic data / provisional schema. Official metric is not yet confirmed.",
        ha="center",
        fontsize=9,
        color="#64748B",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def _display_name(model: str) -> str:
    return model.replace("_", " ").title()


def _annotate_bars(axis: plt.Axes) -> None:
    for bar in axis.patches:
        axis.annotate(
            f"{bar.get_height():.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
