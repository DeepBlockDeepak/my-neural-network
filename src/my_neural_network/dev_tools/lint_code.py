import subprocess


def run():
    """Lints the project's Python code using `ruff`.

    Ensures code adheres to defined standards and catches potential issues.
    """
    try:
        subprocess.run(["ruff", "."], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Linting failed with errors: {e}")
