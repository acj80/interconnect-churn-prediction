from datetime import datetime
from pathlib import Path
import subprocess

from airflow.sdk import DAG, task


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MONITORING_SCRIPT = (
    PROJECT_ROOT
    / "airflow"
    / "scripts"
    / "run_monitoring.py"
)


with DAG(
    dag_id="interconnect_churn_monitoring",
    description=(
        "Automated drift monitoring pipeline "
        "for the Interconnect churn model."
    ),
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=[
        "interconnect",
        "churn",
        "monitoring",
    ],
) as dag:

    @task
    def validate_monitoring_script() -> str:
        if not MONITORING_SCRIPT.exists():
            raise FileNotFoundError(
                f"Monitoring script not found: "
                f"{MONITORING_SCRIPT}"
            )

        return str(
            MONITORING_SCRIPT
        )

    @task
    def run_drift_monitoring(
        script_path: str,
    ) -> str:
        result = subprocess.run(
            [
                "python",
                script_path,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(
                "===== MONITORING STDOUT ====="
            )
            print(
                result.stdout
            )

        if result.stderr:
            print(
                "===== MONITORING STDERR ====="
            )
            print(
                result.stderr
            )

        if result.returncode != 0:
            raise RuntimeError(
                "Monitoring pipeline failed "
                f"with exit code {result.returncode}."
            )

        return (
            "Monitoring pipeline "
            "completed successfully."
        )

    script = (
        validate_monitoring_script()
    )

    run_drift_monitoring(
        script
    )
