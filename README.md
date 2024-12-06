# Simple Neural Network Project

A simple, hand-rolled neural network project for testing and exploration.

### Features

- **Flexible Architecture Configuration:**  
  Easily define custom network layer dimensions via the `layer_dims` parameter.

- **Classification and Regression Tasks:**  
  Train and evaluate the network on various tasks:
  - **Classification:** Tested on datasets like the Breast Cancer and Titanic datasets, providing binary or multi-class classification capabilities.
  - **Regression:** Successfully applied to tasks like predicting apartment rents from the StreetEasy dataset.

- **Multiple Optimizers:**  
  Switch between optimizers like **SGD** and **Adam** without changing your code logic.

- **Dropout Regularization:**  
  Incorporates dropout layers to combat overfitting.

- **Warmup Learning Rate Schedules:**  
  Gradually increases the learning rate at the start of training to stabilize early updates. Configure the warmup steps and scaling factors to improve convergence.

- **Robust Configuration Management with Pydantic:**  
  Employs Pydantic-based configuration (in the `NeuralNetworkConfig`) that validates fields, ensures parameter correctness, and simplifies hyperparameter management.### Features

- **Flexible Architecture Configuration:**  
  Easily define custom network layer dimensions via the `layer_dims` setting. The network supports both shallow and deeper architectures.

- **Classification and Regression Tasks:**  
  Train and evaluate the network on various tasks:
  - **Classification:** Tested on datasets like the Breast Cancer and Titanic datasets, providing binary or multi-class classification capabilities.
  - **Regression:** Successfully applied to tasks like predicting apartment rents from the StreetEasy dataset.

- **Multiple Optimizers:**  
  Switch between optimizers like **SGD** and **Adam** without changing your code logic, allowing you to experiment with different optimization strategies easily.

- **Dropout Regularization:**  
  Incorporate dropout layers to combat overfitting. Control the dropout probability and leverage inverted dropout scaling for consistent training behavior.

- **Warmup Learning Rate Schedules:**  
  Gradually increase the learning rate at the start of training to stabilize early updates. Configure the warmup steps and scaling factors to improve convergence.

- **Robust Configuration Management with Pydantic:**  
  Employ Pydantic-based configuration (e.g., `NeuralNetworkConfig`) that validates fields, ensures parameter correctness, and simplifies hyperparameter management.


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

### Generating Coverage Reports
To generate and view a coverage report, run:
```bash
poetry run coverage
```
This will generate a coverage report and save it as an HTML file in a directory called `coverage_html`. 

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

To perform formatting, linting, and unit-testing sequentially, you can run:
```bash
poetry run all-checks
```

### CI/CD
Continuous Integration is set up with GitHub Actions, which will format, lint, and run tests on the codebase when pushed to the repository.

See [ci.yml](./.github/workflows/ci.yml) for details.
