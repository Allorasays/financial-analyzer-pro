# Setup and Deployment Steps

## Prerequisites

1. **Python 3.11+** installed
2. **Terraform** (optional, for infrastructure deployment)
3. **AWS credentials** configured (for secret upload and deployment)

---

## Step 1: Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

---

## Step 2: Package Lambda Function

**✅ COMPLETED** - Lambda package created:
- File: `lambda/rotation_handler.zip` (13,228 bytes)

To recreate:
```bash
# Windows PowerShell
powershell scripts/package_lambda.ps1

# Linux/Mac
bash scripts/package_lambda.sh

# Or manually
cd lambda
python -c "import zipfile; zf = zipfile.ZipFile('rotation_handler.zip', 'w'); zf.write('rotation_handler.py'); zf.close()"
```

---

## Step 3: Deploy Terraform Infrastructure

**Status**: Terraform files ready, but Terraform not installed locally.

### Install Terraform (if needed)
1. Download from: https://www.terraform.io/downloads
2. Or use package manager (choco, brew, apt)

### Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init

# Review changes
terraform plan

# Apply infrastructure
terraform apply
```

### What gets created:
- KMS key for secret encryption
- Secrets Manager secret
- Lambda function for rotation
- IAM roles and policies
- CloudWatch alarms
- Application IAM role

**Note**: Ensure AWS credentials are configured before running `terraform apply`.

---

## Step 4: Upload Secret to AWS Secrets Manager

### Check AWS Credentials

```bash
python scripts/check_aws.py
```

### Upload Secret

```bash
python scripts/upload_secret.py
```

Or manually:
```bash
aws secretsmanager put-secret-value \
  --secret-id career_planner/college_scorecard_api_key \
  --secret-string file://college_scorecard_api_key.json
```

**Note**: The secret file `college_scorecard_api_key.json` contains the API key.

---

## Step 5: Run Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit

# Run with coverage
pytest --cov=app --cov=lambda --cov=skills --cov=storage --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

Or use Make:
```bash
make test
make test-coverage
```

---

## Step 6: Start Application

```bash
# Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Make
make run
```

### Access the Application

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics
- **Search Colleges**: http://localhost:8000/api/v1/colleges/search?name=MIT

---

## Quick Start (All Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 2. Package Lambda (already done)
# lambda/rotation_handler.zip is ready

# 3. Deploy Terraform (requires Terraform and AWS credentials)
cd terraform
terraform init
terraform apply
cd ..

# 4. Upload secret (requires AWS credentials)
python scripts/upload_secret.py

# 5. Run tests
pytest

# 6. Start application
uvicorn app.main:app --reload
```

---

## Development Workflow

```bash
# Format code
make format
# or
black app/ lambda/ skills/ storage/ tests/

# Lint code
make lint
# or
flake8 app/ lambda/ skills/ storage/ tests/

# Type check
make type-check
# or
mypy app/

# Security check
make security
# or
bandit -r app/ lambda/
detect-secrets scan --baseline .secrets.baseline

# Run all checks
make all-checks
```

---

## Troubleshooting

### ModuleNotFoundError

If you see `ModuleNotFoundError`, install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### AWS Credentials Not Found

Configure AWS credentials:
```bash
aws configure
# Or set environment variables:
# AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY
# AWS_REGION
```

### Terraform Not Found

Install Terraform:
- Windows: Use Chocolatey `choco install terraform`
- Mac: Use Homebrew `brew install terraform`
- Linux: Download from terraform.io/downloads

### Lambda Package Not Found

Recreate the Lambda package:
```bash
cd lambda
python -c "import zipfile; zf = zipfile.ZipFile('rotation_handler.zip', 'w'); zf.write('rotation_handler.py'); zf.close()"
```

---

## Current Status

✅ **Step 1**: Lambda package created (`lambda/rotation_handler.zip`)
⏳ **Step 2**: Terraform files ready (Terraform not installed)
⏳ **Step 3**: Secret file ready (AWS credentials needed)
⏳ **Step 4**: Tests ready (Dependencies need installation)
⏳ **Step 5**: Application ready (Dependencies need installation)

---

## Next Actions

1. Install Python dependencies: `pip install -r requirements.txt -r requirements-dev.txt`
2. Install Terraform (optional, for infrastructure)
3. Configure AWS credentials (for deployment)
4. Run tests: `pytest`
5. Start application: `uvicorn app.main:app --reload`

