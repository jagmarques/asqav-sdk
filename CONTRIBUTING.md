# Contributing to ASQAV SDK

First off, thank you for considering contributing to ASQAV SDK! It's people like you that make open source such a great tool for the community.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open-source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Your First Code Contribution](#your-first-code-contribution)
3. [Styleguides](#styleguides)
    - [Git Commit Messages](#git-commit-messages)
    - [Python Styleguide](#python-styleguide)
4. [Development Environment Setup](#development-environment-setup)
5. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers at [@jagmarques](https://github.com/jagmarques).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please search the [issue tracker](https://github.com/jagmarques/asqav-sdk/issues) to see if the problem has already been reported. If it has, add a comment to the existing issue instead of opening a new one.

When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Describe the behavior you observed** after following the steps and explain precisely what is wrong with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include your Python version and OS.**

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/jagmarques/asqav-sdk/issues). When creating an enhancement suggestion, please:

* **Use a clear and descriptive title.**
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Describe the current behavior** and explain which behavior you expected to see instead and why.
* **Explain why this enhancement would be useful** to most ASQAV SDK users.

### Your First Code Contribution

Unsure where to begin? You can start by looking through [`good-first-issue`](https://github.com/jagmarques/asqav-sdk/labels/good-first-issue) or [`beginner`](https://github.com/jagmarques/asqav-sdk/labels/beginner) issues.

## Development Environment Setup

To start contributing code, you will need a Python development environment (Python 3.10+).

1. **Fork the repository** on GitHub.

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/asqav-sdk.git
   cd asqav-sdk
   ```

3. **Install [uv](https://docs.astral.sh/uv/)** (recommended) and sync dependencies:
   ```bash
   uv sync --all-extras
   ```

   Or use a plain virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[httpx]"
   ```

4. **Run tests** to ensure everything is working:
   ```bash
   uv run python -m pytest tests/
   ```

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature").
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
* Limit the first line to 72 characters or less.
* Reference issues and pull requests liberally after the first line.
* Use [conventional commit](https://www.conventionalcommits.org/) prefixes where applicable:
  * `feat:` — new feature
  * `fix:` — bug fix
  * `docs:` — documentation only
  * `test:` — adding or updating tests
  * `refactor:` — code change that is neither a fix nor a feature

### Python Styleguide

* We follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
* Use **Ruff** for formatting and linting (configured in `pyproject.toml`):
  ```bash
  uv run ruff format src/ tests/
  uv run ruff check src/ tests/
  ```
* Use **mypy** for type checking (strict mode is enabled):
  ```bash
  uv run mypy src/
  ```
* Documentation should follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) docstring format.

## Pull Request Process

1. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** and add tests if applicable.

3. **Ensure all checks pass:**
   ```bash
   uv run python -m pytest tests/
   uv run ruff check src/ tests/
   uv run mypy src/
   ```

4. **Commit your changes:**
   ```bash
   git commit -m 'feat: add some feature'
   ```

5. **Push to your fork:**
   ```bash
   git push origin feat/your-feature-name
   ```

6. **Open a Pull Request** against the `main` branch of the [original repository](https://github.com/jagmarques/asqav-sdk).

7. **Address any feedback** provided by the maintainers in the PR review.

---

Thank you for your contribution!
