# My Neural Network Project

A simple, hand-rolled neural network project for testing and exploration.

### Prerequisites

You need to have **Python 3.10** or newer installed on your system to run this project. This project uses **Poetry** for dependency management.

### Installing

First, clone the repository to your local machine:

```bash
git clone https://github.com/DeepBlockDeepak/my-neural-network.git
cd my-neural-network
```

Then, install the dependencies using Poetry:
```bash
poetry install
```
This will create a virtual environment and install all required dependencies into it.

### Running Tests
This project uses `pytest` for running unit tests. To run the unit tests, use:
```bash
poetry run test
```

### Formatting and Linting
This project uses `isort` and `black` for formatting and `ruff` for linting. You can format and lint the code using the provided custom Poetry scripts:
To Format:
```bash
poetry run format
```
To Lint:
```bash
poetry run lint
```

To perform formatting, linting, and testing sequentially, you can run:
```bash
poetry run all-checks
```

### CI/CD
Continuous Integration is set up with GitHub Actions, which will format, lint, and run tests on the codebase when pushed to the repository.

See [ci.yml](./.github/workflows/ci.yml) for details.




