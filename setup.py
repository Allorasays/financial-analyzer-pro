"""
Setup script for Career Planner Secrets Infrastructure.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="career-planner-secrets-infra",
    version="1.0.0",
    author="Career Planner Team",
    description="AI Career Planner - Secrets Management Infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/career-planner-secrets-infra",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.104.0,<0.105.0",
        "uvicorn[standard]>=0.24.0,<0.25.0",
        "pydantic>=2.5.0,<3.0.0",
        "pydantic-settings>=2.1.0,<3.0.0",
        "httpx>=0.25.0,<0.26.0",
        "boto3>=1.28.0,<2.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "sentence-transformers>=2.2.0,<3.0.0",
        "numpy>=1.24.0,<2.0.0",
        "prometheus_client>=0.19.0,<1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "moto>=4.2.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "bandit>=1.7.0",
            "pre-commit>=3.3.0",
            "detect-secrets>=1.4.0",
        ],
    },
)

