from pathlib import Path
import sys


# ==================================================
# Project path
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==================================================
# Imports that depend on the project root
# ==================================================

import pandas as pd

from monitoring.drift import (
    generate_drift_report_from_profile,
    load_reference_profile,
)


# ==================================================
# Paths
# ==================================================

REFERENCE_PROFILE_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "reference_profile.json"
)

CURRENT_BATCH_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "current_batch.csv"
)

OUTPUT_REPORT_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "drift_report_latest.csv"
)


# ==================================================
# Monitoring pipeline
# ==================================================

def run_monitoring_pipeline() -> Path:
    """
    Execute the operational drift-monitoring pipeline.

    Steps:
    1. Load the persisted reference profile.
    2. Load the current monitoring batch.
    3. Generate PSI-based drift metrics.
    4. Save the latest drift report.
    """

    if not REFERENCE_PROFILE_PATH.exists():
        raise FileNotFoundError(
            "Reference profile not found: "
            f"{REFERENCE_PROFILE_PATH}"
        )

    if not CURRENT_BATCH_PATH.exists():
        raise FileNotFoundError(
            "Current monitoring batch not found: "
            f"{CURRENT_BATCH_PATH}"
        )

    reference_profile = load_reference_profile(
        REFERENCE_PROFILE_PATH
    )

    current_batch = pd.read_csv(
        CURRENT_BATCH_PATH
    )

    drift_report = (
        generate_drift_report_from_profile(
            profile=reference_profile,
            current_df=current_batch,
        )
    )

    OUTPUT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_report.to_csv(
        OUTPUT_REPORT_PATH,
        index=False,
    )

    print("Monitoring pipeline completed.")
    print(f"Report saved to: {OUTPUT_REPORT_PATH}")
    print()
    print(drift_report)

    return OUTPUT_REPORT_PATH


# ==================================================
# Script entry point
# ==================================================

if __name__ == "__main__":
    run_monitoring_pipeline()
