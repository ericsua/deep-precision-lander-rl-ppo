.PHONY: help install install-dev test test-cov format format-check sort sort-check lint type-check clean train evaluate example

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	uv sync

install-dev:  ## Install the package with development dependencies
	uv sync --extra dev

test:  ## Run tests
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov=src/deep_precision_lander_rl_ppo --cov-report=term-missing --cov-report=html

format:  ## Format code with black
	uv run black src tests scripts

format-check:  ## Check code formatting with black
	uv run black --check src tests scripts

sort:  ## Sort imports with isort
	uv run isort src tests scripts

sort-check:  ## Check import sorting with isort
	uv run isort --check-only src tests scripts

lint:  ## Run flake8 linting
	uv run flake8 src tests scripts

type-check:  ## Run mypy type checking
	uv run mypy src/deep_precision_lander_rl_ppo

clean:  ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

train:  ## Run the training script
	uv run python scripts/train.py

evaluate:  ## Run the evaluation script
	uv run python scripts/evaluate.py

example:  ## Run the example script
	uv run python scripts/example.py

all: format sort lint type-check test  ## Run all quality checks and tests
