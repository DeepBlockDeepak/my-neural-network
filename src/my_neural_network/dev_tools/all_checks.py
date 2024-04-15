from .format_code import run as format_run
from .lint_code import run as lint_run
from .run_tests import run as test_run


def run_all():
    """Runs all development checks: formatting, linting, and tests."""
    print("Running format...")
    format_run()

    print("Running lint...")
    lint_run()

    print("Running tests...")
    test_run()


if __name__ == "__main__":
    run_all()
