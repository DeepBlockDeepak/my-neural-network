import subprocess


def run():
    """Formats the project's Python code.

    Uses `isort` to sort import statements, then `black` to format the code.
    """
    subprocess.run(["isort", "."], check=True)
    subprocess.run(["black", "."], check=True)
