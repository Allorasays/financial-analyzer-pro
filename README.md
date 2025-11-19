# AI Career Planner — Secrets & Infra Deliverables

This package uses **AWS Systems Manager Parameter Store** (FREE alternative to Secrets Manager) for secure secret management, with Terraform infrastructure, secure College Scorecard integration, pre-commit secret scanning, GitHub Actions CI workflow, comprehensive tests, monitoring, and helper scripts.

**Why Parameter Store?** Saves ~$2/month vs Secrets Manager while maintaining the same security. Perfect for simple use cases with 1-2 secrets.

IMPORTANT: Do not commit real secret values.

## Setting Up the College Scorecard API Key

The College Scorecard API key has been configured in `college_scorecard_api_key.json`. To upload it to AWS Parameter Store (FREE alternative to Secrets Manager):

### Option 1: Using Python Script (Recommended)

```bash
# Install dependencies if needed
pip install -r requirements.txt

# Upload the parameter (creates or updates)
python scripts/upload_secret_to_ssm.py
```

The script will automatically:
- Create the parameter if it doesn't exist
- Update the parameter if it already exists
- Use the parameter name from config (default: `/career_planner/college_scorecard_api_key`)
- **Cost: FREE** (vs ~$2/month for Secrets Manager)

### Option 2: Using AWS CLI

```bash
# First, ensure Terraform has been applied to create the parameter resource
cd terraform
terraform init
terraform apply

# Then upload the parameter value
cd ..
aws ssm put-parameter \
  --name "/career_planner/college_scorecard_api_key" \
  --value "$(cat college_scorecard_api_key.json | jq -r .COLLEGE_SCORECARD_API_KEY)" \
  --type SecureString \
  --overwrite
```

### Option 3: Using AWS Console

1. Navigate to AWS Systems Manager → Parameter Store in your AWS Console
2. Find or create the parameter named `/career_planner/college_scorecard_api_key`
3. Set type to `SecureString` and paste the API key value
4. Save the parameter

**Note:** The secret file `college_scorecard_api_key.json` is excluded from git via `.gitignore` to prevent accidental commits.

## Running the API locally

```bash
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```

Environment variables can be configured via `.env` and validated through `app/config.py`.

Key endpoints:
- `GET /` – root metadata
- `GET /health` – dependency-aware health check
- `POST|GET /api/v1/colleges/search` – College Scorecard search

## Tests & Quality Gates

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest --cov=app --cov=lambda --cov=skills --cov=storage
black --check app lambda skills storage tests
flake8 app lambda skills storage tests
mypy app lambda skills storage
bandit -r app lambda skills storage
```

The GitHub Actions workflow enforces:
- Linting (black, flake8, mypy)
- Tests with coverage
- Security scanning (detect-secrets, bandit, safety)

## Monitoring & Logging

- Structured JSON logging with secret masking is available via `app/monitoring.py`.
- Prometheus metrics are defined for API requests, secret retrievals, and College Scorecard calls; expose them via FastAPI middleware or a dedicated `/metrics` endpoint as needed.

## Terraform Deployment

```bash
cd terraform
cp backend.tf.example backend.tf   # optional remote state
terraform init
terraform apply
```

**Simplified Infrastructure (Using Parameter Store):**
Provisioned resources:
- Parameter Store parameter (FREE, encrypted with default AWS key)
- Optional KMS key (only if `use_custom_kms = true`)
- Application IAM role & policy
- Optional CloudWatch log group

**Cost:** ~$0.05/month (vs ~$2/month with Secrets Manager)

**Note:** The simplified version (`main_simplified.tf`) is recommended. If you need Secrets Manager with rotation, use the original `main.tf` and ensure `lambda/rotation_handler.zip` is built.

## Security Scanning

- Secret detection uses `.secrets.baseline` (maintain via `detect-secrets scan --baseline .secrets.baseline`).
- Logs automatically mask high-entropy strings.
- CI runs Bandit and Safety; run locally with:

```bash
detect-secrets scan --baseline .secrets.baseline
bandit -r app lambda skills storage -ll
safety check --file requirements.txt --full-report
```
