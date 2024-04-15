import subprocess


def run():
    """Runs all unittests within the `tests` directory.

    Utilizes Python's built-in unittest discovery to run all tests.
    """
    subprocess.run(["python", "-m", "unittest", "discover", "-s", "tests"], check=True)
