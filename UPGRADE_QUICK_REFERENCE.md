# Quick Upgrade Reference

## 🔥 Critical Upgrades (Do First)

### 1. Dependency Updates

```bash
# Update requirements.txt
fastapi>=0.115.0,<0.116.0  # was 0.104.0
pydantic>=2.9.0,<3.0.0     # was 2.5.0
uvicorn[standard]>=0.32.0,<0.33.0  # was 0.24.0
httpx>=0.28.0,<0.29.0      # was 0.25.0
boto3>=1.35.0,<2.0.0       # was 1.28.0
```

### 2. Python Version
- Current: 3.11
- Target: 3.12 or 3.13
- Update: `.github/workflows/ci.yml`, `pyproject.toml`, `setup.py`

### 3. Security
- Add API authentication
- Implement rate limiting (slowapi is installed but not used)
- Add request validation

### 4. Missing Features
- Database models (SQLAlchemy unused)
- Caching (Redis or in-memory)
- Request/response middleware
- Error tracking (Sentry)

---

## 📦 Immediate Commands

```bash
# Update dependencies
pip install --upgrade fastapi uvicorn pydantic pydantic-settings httpx boto3

# Update dev dependencies
pip install --upgrade pytest pytest-cov pytest-asyncio black flake8 mypy

# Test compatibility
pytest

# Check for vulnerabilities
pip-audit
safety check
```

---

## 🎯 Priority Order

1. **Security** - Authentication, rate limiting, input validation
2. **Dependencies** - FastAPI, Pydantic, Python version
3. **Features** - Database, caching, monitoring
4. **Infrastructure** - Docker, Kubernetes, CI/CD
5. **Quality** - Tests, documentation, performance

---

See `UPGRADE_PLAN.md` for detailed upgrade instructions.

