# Base Image
FROM python:3.11-slim

# Argument for environment (default to development)
ARG YOUR_ENV=development
ARG POETRY_VERSION=1.7.1

# Environment Variables for Python and Poetry
ENV YOUR_ENV=${YOUR_ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR="/var/cache/pypoetry" \
    POETRY_HOME="/usr/local/poetry" \
    PATH="${POETRY_HOME}/bin:$PATH" \
    PYTHONPATH="/code/src" 

# Install curl and Poetry
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry && \
    poetry --version  # Verify Poetry installation

# Set working directory
WORKDIR /code

# Copy dependency files
COPY pyproject.toml poetry.lock /code/

# Install dependencies based on the environment
RUN /usr/local/bin/poetry install $(test "$YOUR_ENV" = production && echo "--only=main") --no-interaction --no-ansi

# Copy the rest of the project files
COPY . /code

# Ensures all entry points and scripts in pyproject.toml are correctly installed and registered
RUN poetry install --no-interaction --no-ansi --with dev