import subprocess


def run():
    """Runs all tests within the `tests` directory with coverage reporting."""
    subprocess.run(
        [
            "poetry",
            "run",
            "pytest",
            "tests",
            "--cov=my_neural_network",
            "--cov-report=term",
            "--cov-report=html:coverage_html",
            "--cov-fail-under=80",
        ],
        check=True,
    )
