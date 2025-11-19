.PHONY: help install install-dev test lint format type-check security clean run deploy-terraform package-lambda

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test: ## Run tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit

test-integration: ## Run integration tests only
	pytest tests/integration

test-coverage: ## Run tests with coverage report
	pytest --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint: ## Run linting checks
	flake8 app/ lambda/ skills/ storage/ tests/ --max-line-length=120 --exclude=__pycache__,*.pyc
	black --check app/ lambda/ skills/ storage/ tests/

format: ## Format code with black
	black app/ lambda/ skills/ storage/ tests/

type-check: ## Run type checking with mypy
	mypy app/ --ignore-missing-imports --no-strict-optional

security: ## Run security checks
	detect-secrets scan --baseline .secrets.baseline
	bandit -r app/ lambda/ -f json -o bandit-report.json || true

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true
	rm -f .coverage coverage.xml bandit-report.json

run: ## Run the FastAPI application
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

package-lambda: ## Package Lambda function for deployment
	cd lambda && zip -r rotation_handler.zip rotation_handler.py
	@echo "Lambda package created: lambda/rotation_handler.zip"

deploy-terraform: ## Deploy Terraform infrastructure
	cd terraform && terraform init && terraform plan && terraform apply

terraform-plan: ## Plan Terraform changes
	cd terraform && terraform init && terraform plan

terraform-destroy: ## Destroy Terraform infrastructure
	cd terraform && terraform destroy

upload-secret: ## Upload secret to AWS Secrets Manager
	python scripts/upload_secret.py

pre-commit: ## Run pre-commit checks
	pre-commit run --all-files

all-checks: lint type-check security test ## Run all checks (lint, type-check, security, test)

