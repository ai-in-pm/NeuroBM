# NeuroBM Makefile
# Comprehensive build and development automation

.PHONY: help install install-dev test test-fast test-slow lint format type-check clean docs serve-docs validate scaffold sync watch build publish

# Default target
help:
	@echo "NeuroBM Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install       Install package in development mode"
	@echo "  install-dev   Install with development dependencies"
	@echo "  install-all   Install with all optional dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run flake8 linting"
	@echo "  type-check    Run mypy type checking"
	@echo "  check         Run all code quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run fast tests only"
	@echo "  test-slow     Run slow/integration tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  test-core     Run core functionality test"
	@echo ""
	@echo "Project Management:"
	@echo "  validate      Validate project structure"
	@echo "  scaffold      Initialize/update project scaffolding"
	@echo "  sync          Sync project files and structure"
	@echo "  watch         Watch for changes and auto-validate"
	@echo "  clean         Clean temporary files and caches"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          Build documentation"
	@echo "  serve-docs    Serve documentation locally"
	@echo ""
	@echo "Training and Experiments:"
	@echo "  train-base    Train base cognitive model"
	@echo "  train-ptsd    Train PTSD model"
	@echo "  train-autism  Train autism model"
	@echo "  train-ai      Train AI-reliance model"
	@echo ""
	@echo "Build and Release:"
	@echo "  build         Build package for distribution"
	@echo "  publish       Publish to PyPI (requires credentials)"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# Code quality targets
format:
	black neurobm/ scripts/ tests/
	isort neurobm/ scripts/ tests/

lint:
	flake8 neurobm/ scripts/ tests/

type-check:
	mypy neurobm/ scripts/

check: format lint type-check
	@echo "All code quality checks passed!"

# Testing targets
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-slow:
	pytest tests/ -v -m "slow"

test-cov:
	pytest tests/ --cov=neurobm --cov-report=html --cov-report=term

test-core:
	python test_neurobm_core.py

# Project management targets
validate:
	python scripts/neurobm_scaffold.py validate

scaffold:
	python scripts/neurobm_scaffold.py init

sync:
	python scripts/neurobm_scaffold.py sync

watch:
	python scripts/neurobm_scaffold.py watch

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf runs/ archives/ *.prof *.lprof *.mprof

# Documentation targets
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Training targets
train-base:
	python scripts/train.py --exp=base --hidden=256 --k=1 --epochs=50

train-ptsd:
	python scripts/train.py --exp=ptsd --hidden=512 --k=10 --pcd --epochs=100

train-autism:
	python scripts/train.py --exp=autism --hidden=256 --patience=15

train-ai:
	python scripts/train.py --exp=ai_reliance --hidden=256 --k=5 --epochs=75

train-ptsd-pm:
	python scripts/train.py --exp=ptsd_pm --hidden=20 --k=3 --epochs=100 --lr=0.005

# Build and release targets
build:
	python -m build

publish:
	python -m twine upload dist/*

# Development workflow shortcuts
dev-setup: install-dev
	pre-commit install
	python scripts/neurobm_scaffold.py init
	@echo "Development environment ready!"

quick-test: test-core test-fast
	@echo "Quick validation complete!"

full-check: check test validate
	@echo "Full project validation complete!"

# Continuous integration simulation
ci: install-dev check test-cov validate
	@echo "CI pipeline simulation complete!"