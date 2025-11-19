# Contributing to Career Planner Secrets Infrastructure

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd career_planner_secrets_infra
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   # or
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Setup pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add type hints to all functions
   - Add docstrings to all public functions/classes
   - Write tests for new functionality

3. **Run tests and checks**
   ```bash
   make all-checks
   # or individually:
   make lint
   make type-check
   make security
   make test
   ```

4. **Format your code**
   ```bash
   make format
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Description of changes"
   ```
   - Use clear, descriptive commit messages
   - Reference issue numbers if applicable

6. **Push and create a Pull Request**

## Code Style

- **Python**: Follow PEP 8 style guide
- **Line length**: 120 characters maximum
- **Type hints**: Required for all function signatures
- **Docstrings**: Required for all public functions/classes (Google style)
- **Formatting**: Use Black (configured in `pyproject.toml`)

## Testing

- Write unit tests for all new functionality
- Aim for >80% code coverage
- Run tests before committing: `make test`
- Use pytest fixtures for test setup/teardown
- Mock external services (AWS, APIs) in tests

## Security

- Never commit secrets or API keys
- Use AWS Secrets Manager for sensitive data
- Run security scans: `make security`
- Review `.secrets.baseline` before committing

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update API documentation if endpoints change
- Keep CHANGELOG.md updated (if maintained)

## Terraform Changes

- Always run `terraform fmt` before committing
- Test with `terraform plan` before applying
- Update variables with descriptions
- Document any infrastructure changes in PR description

## Pull Request Process

1. Ensure all tests pass
2. Ensure code is formatted and linted
3. Update documentation as needed
4. Request review from maintainers
5. Address review feedback
6. Maintainers will merge when approved

## Reporting Issues

- Use GitHub Issues for bug reports
- Include steps to reproduce
- Include error messages and logs
- Describe expected vs actual behavior

## Questions?

Feel free to open an issue for questions or reach out to maintainers.

