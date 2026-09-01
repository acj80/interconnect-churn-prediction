import json
from pathlib import Path
import numpy as np
import pandas as pd


EPSILON = 1e-6


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI)
    for numerical distributions.
    """

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.linspace(
        0,
        100,
        bins + 1,
    )

    cut_points = np.percentile(
        expected,
        breakpoints,
    )

    cut_points = np.unique(cut_points)

    if len(cut_points) < 2:
        return 0.0

    expected_counts, _ = np.histogram(
        expected,
        bins=cut_points,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=cut_points,
    )

    expected_pct = (
        expected_counts
        / max(expected_counts.sum(), 1)
    )

    actual_pct = (
        actual_counts
        / max(actual_counts.sum(), 1)
    )

    expected_pct = np.clip(
        expected_pct,
        EPSILON,
        None,
    )

    actual_pct = np.clip(
        actual_pct,
        EPSILON,
        None,
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(
            actual_pct / expected_pct
        )
    )

    return float(psi)


def calculate_categorical_psi(
    expected: pd.Series,
    actual: pd.Series,
) -> float:
    """
    Calculate PSI for categorical variables.
    """

    expected = expected.astype(str)
    actual = actual.astype(str)

    categories = sorted(
        set(expected.unique())
        | set(actual.unique())
    )

    expected_dist = (
        expected.value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
    )

    actual_dist = (
        actual.value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
    )

    expected_pct = np.clip(
        expected_dist.values,
        EPSILON,
        None,
    )

    actual_pct = np.clip(
        actual_dist.values,
        EPSILON,
        None,
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(
            actual_pct / expected_pct
        )
    )

    return float(psi)


def classify_drift(
    psi: float,
) -> str:
    """
    Interpret PSI using common monitoring thresholds.
    """

    if psi < 0.10:
        return "stable"

    if psi < 0.25:
        return "moderate"

    return "significant"


def monitor_numeric_feature(
    reference: pd.Series,
    current: pd.Series,
) -> dict:
    """
    Monitor one numerical feature.
    """

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    psi = calculate_psi(
        reference.values,
        current.values,
    )

    reference_mean = float(
        reference.mean()
    )

    current_mean = float(
        current.mean()
    )

    mean_change_pct = (
        (
            current_mean - reference_mean
        )
        / reference_mean
        * 100
        if reference_mean != 0
        else 0.0
    )

    return {
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "mean_change_pct": float(
            mean_change_pct
        ),
        "psi": psi,
        "drift_status": classify_drift(
            psi
        ),
    }


def monitor_categorical_feature(
    reference: pd.Series,
    current: pd.Series,
) -> dict:
    """
    Monitor one categorical feature.
    """

    psi = calculate_categorical_psi(
        reference,
        current,
    )

    return {
        "psi": psi,
        "drift_status": classify_drift(
            psi
        ),
    }


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """
    Generate a complete drift report.
    """

    rows = []

    for feature in numeric_features:

        result = monitor_numeric_feature(
            reference_df[feature],
            current_df[feature],
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "psi": result["psi"],
                "drift_status": result[
                    "drift_status"
                ],
                "reference_mean": result[
                    "reference_mean"
                ],
                "current_mean": result[
                    "current_mean"
                ],
                "mean_change_pct": result[
                    "mean_change_pct"
                ],
            }
        )

    for feature in categorical_features:

        result = monitor_categorical_feature(
            reference_df[feature],
            current_df[feature],
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "psi": result["psi"],
                "drift_status": result[
                    "drift_status"
                ],
                "reference_mean": np.nan,
                "current_mean": np.nan,
                "mean_change_pct": np.nan,
            }
        )

    report = pd.DataFrame(rows)

    return report.sort_values(
        by="psi",
        ascending=False,
    ).reset_index(drop=True)


def build_reference_profile(
    reference_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    bins: int = 10,
) -> dict:
    """
    Build a persistent reference profile for drift monitoring.

    The profile contains aggregated distributions only.
    Raw customer records are not stored.
    """

    profile = {
        "numeric_features": {},
        "categorical_features": {},
    }

    # ----------------------------------------------
    # Numerical features
    # ----------------------------------------------

    for feature in numeric_features:

        series = pd.to_numeric(
            reference_df[feature],
            errors="coerce",
        ).dropna()

        percentiles = np.linspace(
            0,
            100,
            bins + 1,
        )

        bin_edges = np.percentile(
            series,
            percentiles,
        )

        bin_edges = np.unique(
            bin_edges
        )

        if len(bin_edges) < 2:
            bin_edges = np.array(
                [
                    float(series.min()),
                    float(series.max()) + EPSILON,
                ]
            )

        counts, _ = np.histogram(
            series,
            bins=bin_edges,
        )

        proportions = (
            counts
            / max(counts.sum(), 1)
        )

        profile["numeric_features"][
            feature
        ] = {
            "mean": float(
                series.mean()
            ),
            "std": float(
                series.std()
            ),
            "min": float(
                series.min()
            ),
            "max": float(
                series.max()
            ),
            "bin_edges": [
                float(value)
                for value in bin_edges
            ],
            "proportions": [
                float(value)
                for value in proportions
            ],
        }

    # ----------------------------------------------
    # Categorical features
    # ----------------------------------------------

    for feature in categorical_features:

        series = (
            reference_df[feature]
            .astype(str)
        )

        distribution = (
            series.value_counts(
                normalize=True
            )
        )

        profile["categorical_features"][
            feature
        ] = {
            category: float(
                proportion
            )
            for category, proportion
            in distribution.items()
        }

    return profile


def save_reference_profile(
    profile: dict,
    output_path: str | Path,
) -> None:
    """
    Save a monitoring reference profile as JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            profile,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_reference_profile(
    profile_path: str | Path,
) -> dict:
    """
    Load a monitoring reference profile from JSON.
    """

    profile_path = Path(
        profile_path
    )

    with profile_path.open(
        encoding="utf-8",
    ) as file:
        return json.load(file)


def calculate_psi_from_proportions(
    expected_proportions,
    actual_proportions,
) -> float:
    """
    Calculate PSI directly from two distributions
    expressed as proportions.
    """

    expected_pct = np.asarray(
        expected_proportions,
        dtype=float,
    )

    actual_pct = np.asarray(
        actual_proportions,
        dtype=float,
    )

    expected_pct = np.clip(
        expected_pct,
        EPSILON,
        None,
    )

    actual_pct = np.clip(
        actual_pct,
        EPSILON,
        None,
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(
            actual_pct / expected_pct
        )
    )

    return float(psi)


def monitor_numeric_from_profile(
    reference_profile: dict,
    current: pd.Series,
) -> dict:
    """
    Compare a current numerical feature against
    its stored reference profile.
    """

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if current.empty:
        raise ValueError(
            "Current numerical feature contains "
            "no valid observations."
        )

    reference_proportions = np.asarray(
        reference_profile["proportions"],
        dtype=float,
    )

    bin_edges = np.asarray(
        reference_profile["bin_edges"],
        dtype=float,
    )

    # Extend the first and last bins so values outside
    # the original training range are also captured.
    monitoring_edges = bin_edges.copy()

    monitoring_edges[0] = -np.inf
    monitoring_edges[-1] = np.inf

    current_counts, _ = np.histogram(
        current.values,
        bins=monitoring_edges,
    )

    current_proportions = (
        current_counts
        / max(current_counts.sum(), 1)
    )

    psi = calculate_psi_from_proportions(
        reference_proportions,
        current_proportions,
    )

    reference_mean = float(
        reference_profile["mean"]
    )

    current_mean = float(
        current.mean()
    )

    mean_change_pct = (
        (
            current_mean - reference_mean
        )
        / reference_mean
        * 100
        if reference_mean != 0
        else 0.0
    )

    return {
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "mean_change_pct": float(
            mean_change_pct
        ),
        "psi": psi,
        "drift_status": classify_drift(
            psi
        ),
    }


def monitor_categorical_from_profile(
    reference_profile: dict,
    current: pd.Series,
) -> dict:
    """
    Compare a current categorical feature against
    its stored reference distribution.
    """

    current = (
        current
        .astype(str)
    )

    current_distribution = (
        current.value_counts(
            normalize=True
        )
    )

    categories = sorted(
        set(reference_profile.keys())
        | set(current_distribution.index)
    )

    expected_proportions = [
        float(
            reference_profile.get(
                category,
                0.0,
            )
        )
        for category in categories
    ]

    actual_proportions = [
        float(
            current_distribution.get(
                category,
                0.0,
            )
        )
        for category in categories
    ]

    psi = calculate_psi_from_proportions(
        expected_proportions,
        actual_proportions,
    )

    return {
        "psi": psi,
        "drift_status": classify_drift(
            psi
        ),
    }


def generate_drift_report_from_profile(
    profile: dict,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a drift report using a persisted
    reference profile and a current data batch.
    """

    rows = []

    numeric_profiles = profile.get(
        "numeric_features",
        {},
    )

    categorical_profiles = profile.get(
        "categorical_features",
        {},
    )

    required_features = (
        set(numeric_profiles)
        | set(categorical_profiles)
    )

    missing_features = (
        required_features
        - set(current_df.columns)
    )

    if missing_features:
        raise ValueError(
            "Current batch is missing required "
            f"features: {sorted(missing_features)}"
        )

    # ----------------------------------------------
    # Numerical features
    # ----------------------------------------------

    for feature, feature_profile in (
        numeric_profiles.items()
    ):

        result = monitor_numeric_from_profile(
            feature_profile,
            current_df[feature],
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "psi": result["psi"],
                "drift_status": result[
                    "drift_status"
                ],
                "reference_mean": result[
                    "reference_mean"
                ],
                "current_mean": result[
                    "current_mean"
                ],
                "mean_change_pct": result[
                    "mean_change_pct"
                ],
            }
        )

    # ----------------------------------------------
    # Categorical features
    # ----------------------------------------------

    for feature, feature_profile in (
        categorical_profiles.items()
    ):

        result = (
            monitor_categorical_from_profile(
                feature_profile,
                current_df[feature],
            )
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "psi": result["psi"],
                "drift_status": result[
                    "drift_status"
                ],
                "reference_mean": np.nan,
                "current_mean": np.nan,
                "mean_change_pct": np.nan,
            }
        )

    report = pd.DataFrame(
        rows
    )

    return report.sort_values(
        by="psi",
        ascending=False,
    ).reset_index(
        drop=True
    )


def calculate_psi_from_proportions(
    expected_proportions,
    actual_proportions,
) -> float:
    """
    Calculate PSI directly from two distributions
    expressed as proportions.
    """

    expected_pct = np.asarray(
        expected_proportions,
        dtype=float,
    )

    actual_pct = np.asarray(
        actual_proportions,
        dtype=float,
    )

    expected_pct = np.clip(
        expected_pct,
        EPSILON,
        None,
    )

    actual_pct = np.clip(
        actual_pct,
        EPSILON,
        None,
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(
            actual_pct / expected_pct
        )
    )

    return float(psi)


def monitor_numeric_from_profile(
    reference_profile: dict,
    current: pd.Series,
) -> dict:
    """
    Compare a current numerical feature against
    its stored reference profile.
    """

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if current.empty:
        raise ValueError(
            "Current numerical feature contains "
            "no valid observations."
        )

    reference_proportions = np.asarray(
        reference_profile["proportions"],
        dtype=float,
    )

    bin_edges = np.asarray(
        reference_profile["bin_edges"],
        dtype=float,
    )

    monitoring_edges = bin_edges.copy()

    monitoring_edges[0] = -np.inf
    monitoring_edges[-1] = np.inf

    current_counts, _ = np.histogram(
        current.values,
        bins=monitoring_edges,
    )

    current_proportions = (
        current_counts
        / max(current_counts.sum(), 1)
    )

    psi = calculate_psi_from_proportions(
        reference_proportions,
        current_proportions,
    )

    reference_mean = float(
        reference_profile["mean"]
    )

    current_mean = float(
        current.mean()
    )

    mean_change_pct = (
        (
            current_mean - reference_mean
        )
        / reference_mean
        * 100
        if reference_mean != 0
        else 0.0
    )

    return {
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "mean_change_pct": float(
            mean_change_pct
        ),
        "psi": psi,
        "drift_status": classify_drift(
            psi
        ),
    }


def monitor_categorical_from_profile(
    reference_profile: dict,
    current: pd.Series,
) -> dict:
    """
    Compare a current categorical feature against
    its stored reference distribution.
    """

    current = current.astype(str)

    current_distribution = (
        current.value_counts(
            normalize=True
        )
    )

    categories = sorted(
        set(reference_profile.keys())
        | set(current_distribution.index)
    )

    expected_proportions = [
        float(
            reference_profile.get(
                category,
                0.0,
            )
        )
        for category in categories
    ]

    actual_proportions = [
        float(
            current_distribution.get(
                category,
                0.0,
            )
        )
        for category in categories
    ]

    psi = calculate_psi_from_proportions(
        expected_proportions,
        actual_proportions,
    )

    return {
        "psi": psi,
        "drift_status": classify_drift(
            psi
        ),
    }


def generate_drift_report_from_profile(
    profile: dict,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a drift report using a persisted
    reference profile and a current data batch.
    """

    rows = []

    numeric_profiles = profile.get(
        "numeric_features",
        {},
    )

    categorical_profiles = profile.get(
        "categorical_features",
        {},
    )

    required_features = (
        set(numeric_profiles)
        | set(categorical_profiles)
    )

    missing_features = (
        required_features
        - set(current_df.columns)
    )

    if missing_features:
        raise ValueError(
            "Current batch is missing required "
            f"features: {sorted(missing_features)}"
        )

    for feature, feature_profile in (
        numeric_profiles.items()
    ):

        result = monitor_numeric_from_profile(
            feature_profile,
            current_df[feature],
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "psi": result["psi"],
                "drift_status": result[
                    "drift_status"
                ],
                "reference_mean": result[
                    "reference_mean"
                ],
                "current_mean": result[
                    "current_mean"
                ],
                "mean_change_pct": result[
                    "mean_change_pct"
                ],
            }
        )

    for feature, feature_profile in (
        categorical_profiles.items()
    ):

        result = (
            monitor_categorical_from_profile(
                feature_profile,
                current_df[feature],
            )
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "psi": result["psi"],
                "drift_status": result[
                    "drift_status"
                ],
                "reference_mean": np.nan,
                "current_mean": np.nan,
                "mean_change_pct": np.nan,
            }
        )

    report = pd.DataFrame(
        rows
    )

    return report.sort_values(
        by="psi",
        ascending=False,
    ).reset_index(
        drop=True
    )
