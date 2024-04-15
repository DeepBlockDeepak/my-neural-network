import subprocess


def run():
    """Runs all tests within the `tests` directory.

    Utilizes pytest to run all tests.
    """
    subprocess.run(["poetry", "run", "pytest", "tests"], check=True)
