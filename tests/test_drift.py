import numpy as np
import pandas as pd

from monitoring.drift import (
    build_reference_profile,
    calculate_categorical_psi,
    calculate_psi,
    classify_drift,
    generate_drift_report,
    generate_drift_report_from_profile,
    load_reference_profile,
    save_reference_profile,
)


def test_numeric_psi_same_distribution():
    expected = np.array(
        [10, 20, 30, 40, 50] * 20
    )

    actual = expected.copy()

    psi = calculate_psi(
        expected,
        actual,
    )

    assert psi < 0.01


def test_numeric_psi_detects_shift():
    expected = np.array(
        [10, 20, 30, 40, 50] * 20
    )

    actual = np.array(
        [60, 70, 80, 90, 100] * 20
    )

    psi = calculate_psi(
        expected,
        actual,
    )

    assert psi >= 0.25


def test_categorical_psi_same_distribution():
    expected = pd.Series(
        ["A", "A", "B", "B"] * 25
    )

    actual = expected.copy()

    psi = calculate_categorical_psi(
        expected,
        actual,
    )

    assert psi < 0.01


def test_categorical_psi_detects_shift():
    expected = pd.Series(
        ["A"] * 90
        + ["B"] * 10
    )

    actual = pd.Series(
        ["A"] * 10
        + ["B"] * 90
    )

    psi = calculate_categorical_psi(
        expected,
        actual,
    )

    assert psi >= 0.25


def test_classify_drift():
    assert classify_drift(0.05) == "stable"
    assert classify_drift(0.15) == "moderate"
    assert classify_drift(0.30) == "significant"


def test_generate_drift_report():
    reference_df = pd.DataFrame(
        {
            "MonthlyCharges": [
                20,
                30,
                40,
                50,
                60,
            ]
            * 20,
            "Type": [
                "Month-to-month",
                "One year",
            ]
            * 50,
        }
    )

    current_df = reference_df.copy()

    report = generate_drift_report(
        reference_df=reference_df,
        current_df=current_df,
        numeric_features=[
            "MonthlyCharges",
        ],
        categorical_features=[
            "Type",
        ],
    )

    assert len(report) == 2

    assert set(
        report["feature"]
    ) == {
        "MonthlyCharges",
        "Type",
    }

    assert (
        report["drift_status"]
        == "stable"
    ).all()


def test_build_reference_profile():
    reference_df = pd.DataFrame(
        {
            "MonthlyCharges": [
                10,
                20,
                30,
                40,
                50,
            ]
            * 20,
            "Type": [
                "Month-to-month",
                "One year",
            ]
            * 50,
        }
    )

    profile = build_reference_profile(
        reference_df=reference_df,
        numeric_features=[
            "MonthlyCharges",
        ],
        categorical_features=[
            "Type",
        ],
    )

    assert (
        "MonthlyCharges"
        in profile["numeric_features"]
    )

    assert (
        "Type"
        in profile["categorical_features"]
    )

    assert (
        profile["numeric_features"]
        ["MonthlyCharges"]
        ["mean"]
        == 30.0
    )

    type_distribution = (
        profile["categorical_features"]
        ["Type"]
    )

    assert abs(
        type_distribution[
            "Month-to-month"
        ]
        - 0.5
    ) < 1e-12

    assert abs(
        type_distribution[
            "One year"
        ]
        - 0.5
    ) < 1e-12


def test_reference_profile_roundtrip(
    tmp_path,
):
    reference_df = pd.DataFrame(
        {
            "MonthlyCharges": [
                10,
                20,
                30,
                40,
                50,
            ],
            "Type": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "One year",
                "Two year",
            ],
        }
    )

    profile = build_reference_profile(
        reference_df=reference_df,
        numeric_features=[
            "MonthlyCharges",
        ],
        categorical_features=[
            "Type",
        ],
    )

    profile_path = (
        tmp_path
        / "reference_profile.json"
    )

    save_reference_profile(
        profile,
        profile_path,
    )

    loaded_profile = (
        load_reference_profile(
            profile_path
        )
    )

    assert loaded_profile == profile

# ==================================================
# Tests para generate_drift_report_from_profile
# ==================================================


def test_report_from_profile_same_distribution():
    reference_df = pd.DataFrame(
        {
            "MonthlyCharges": [
                20,
                30,
                40,
                50,
                60,
            ]
            * 20,
            "Type": [
                "Month-to-month",
                "One year",
            ]
            * 50,
        }
    )

    profile = build_reference_profile(
        reference_df=reference_df,
        numeric_features=[
            "MonthlyCharges",
        ],
        categorical_features=[
            "Type",
        ],
    )

    current_df = reference_df.copy()

    report = (
        generate_drift_report_from_profile(
            profile=profile,
            current_df=current_df,
        )
    )

    assert len(report) == 2

    assert (
        report["drift_status"]
        == "stable"
    ).all()


def test_report_from_profile_detects_drift():
    reference_df = pd.DataFrame(
        {
            "MonthlyCharges": [
                20,
                30,
                40,
                50,
                60,
            ]
            * 20,
            "Type": (
                ["Month-to-month"] * 80
                + ["One year"] * 20
            ),
        }
    )

    profile = build_reference_profile(
        reference_df=reference_df,
        numeric_features=[
            "MonthlyCharges",
        ],
        categorical_features=[
            "Type",
        ],
    )

    current_df = pd.DataFrame(
        {
            "MonthlyCharges": [
                100,
                110,
                120,
                130,
                140,
            ]
            * 20,
            "Type": (
                ["Month-to-month"] * 10
                + ["One year"] * 90
            ),
        }
    )

    report = (
        generate_drift_report_from_profile(
            profile=profile,
            current_df=current_df,
        )
    )

    statuses = dict(
        zip(
            report["feature"],
            report["drift_status"],
        )
    )

    assert (
        statuses["MonthlyCharges"]
        == "significant"
    )

    assert (
        statuses["Type"]
        == "significant"
    )


def test_report_from_profile_missing_feature():
    reference_df = pd.DataFrame(
        {
            "MonthlyCharges": [
                20,
                30,
                40,
                50,
                60,
            ],
            "Type": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "One year",
                "Two year",
            ],
        }
    )

    profile = build_reference_profile(
        reference_df=reference_df,
        numeric_features=[
            "MonthlyCharges",
        ],
        categorical_features=[
            "Type",
        ],
    )

    current_df = pd.DataFrame(
        {
            "MonthlyCharges": [
                30,
                40,
                50,
            ]
        }
    )

    try:
        generate_drift_report_from_profile(
            profile=profile,
            current_df=current_df,
        )

        assert False

    except ValueError as error:
        assert (
            "missing required features"
            in str(error)
        )
