from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

ALPHA = 0.05
POWER = 0.80
BASELINE_RETURN_RATE = 0.954216
BASELINE_DISENGAGEMENT_RATE = 1 - BASELINE_RETURN_RATE


def two_proportion_sample_size(
    baseline_rate: float,
    treatment_rate: float,
    alpha: float = ALPHA,
    power: float = POWER,
) -> int:
    """Approximate per-group sample size for a two-sided two-proportion z-test."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    pooled_rate = (baseline_rate + treatment_rate) / 2
    numerator = (
        z_alpha * (2 * pooled_rate * (1 - pooled_rate)) ** 0.5
        + z_power
        * (
            baseline_rate * (1 - baseline_rate)
            + treatment_rate * (1 - treatment_rate)
        )
        ** 0.5
    ) ** 2
    effect = abs(treatment_rate - baseline_rate)
    return int(round(numerator / effect**2 + 0.5))


def build_sample_size_table() -> pd.DataFrame:
    rows = []
    for absolute_lift in [0.0025, 0.005, 0.01, 0.015, 0.02]:
        treatment_rate = min(BASELINE_RETURN_RATE + absolute_lift, 0.999)
        per_group = two_proportion_sample_size(BASELINE_RETURN_RATE, treatment_rate)
        rows.append(
            {
                "baseline_return_rate": BASELINE_RETURN_RATE,
                "absolute_lift": absolute_lift,
                "treatment_return_rate": treatment_rate,
                "relative_lift": absolute_lift / BASELINE_RETURN_RATE,
                "alpha": ALPHA,
                "power": POWER,
                "required_users_per_group": per_group,
                "required_users_total": per_group * 2,
            }
        )
    return pd.DataFrame(rows)


def experiment_design_summary() -> pd.DataFrame:
    rows = [
        {
            "design_element": "Business objective",
            "choice": "Test whether improved personalization increases observed user retention.",
            "rationale": "Offline recommendation gains do not prove retention impact.",
        },
        {
            "design_element": "Control",
            "choice": "Current production recommendation experience.",
            "rationale": "Represents the status quo the product team would otherwise keep.",
        },
        {
            "design_element": "Treatment",
            "choice": "Personalized feed strategy informed by collaborative filtering and disengagement-risk segments.",
            "rationale": "Phase 6 showed SVD improves offline ranking, especially for users with sufficient history.",
        },
        {
            "design_element": "Unit of randomization",
            "choice": "User",
            "rationale": "Retention is user-level, and randomizing events could expose the same user to both experiences.",
        },
        {
            "design_element": "Eligibility",
            "choice": "Active users with enough prior interaction history for personalization.",
            "rationale": "SVD performed better for higher-activity users and struggles with cold start.",
        },
        {
            "design_element": "Primary metric",
            "choice": "7-day observed return rate after experiment assignment.",
            "rationale": "Directly matches the retention objective and is easy to explain.",
        },
        {
            "design_element": "Secondary metrics",
            "choice": "Long-view rate, sessions per user, watch time per active user, and recommendation Hit Rate@K.",
            "rationale": "Helps diagnose how the treatment affects engagement quality.",
        },
        {
            "design_element": "Guardrail metrics",
            "choice": "Hate rate, short-view rate, app exits if available, and creator/content concentration.",
            "rationale": "Protects against optimizing retention while harming user experience or ecosystem health.",
        },
        {
            "design_element": "Hypothesis",
            "choice": "Null: treatment return rate equals control. Alternative: treatment return rate differs from control.",
            "rationale": "A two-sided test avoids assuming the personalization change cannot hurt.",
        },
    ]
    return pd.DataFrame(rows)


def save_sample_size_figure(sample_sizes: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        sample_sizes["absolute_lift"] * 100,
        sample_sizes["required_users_total"],
        marker="o",
    )
    ax.set_title("Experiment Sample Size vs Detectable Retention Lift")
    ax.set_xlabel("Minimum detectable absolute lift, percentage points")
    ax.set_ylabel("Required total users")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase8_sample_size_by_mde.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    design = experiment_design_summary()
    sample_sizes = build_sample_size_table()

    design.to_csv(REPORTS / "phase8_experiment_design_summary.csv", index=False)
    sample_sizes.to_csv(REPORTS / "phase8_sample_size_estimates.csv", index=False)
    save_sample_size_figure(sample_sizes)

    print("\n=== Experiment design summary ===")
    print(design.to_string(index=False))

    print("\n=== Sample size estimates ===")
    print(sample_sizes.to_string(index=False))

    print("\nSaved Phase 8 experiment design tables and sample-size figure.")


if __name__ == "__main__":
    main()
