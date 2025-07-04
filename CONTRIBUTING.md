# Contributing to Pinecone RAG Chatbot

Thank you for your interest in contributing to the Pinecone RAG Chatbot project! 

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/pinecone.git
   cd pinecone
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install dev dependencies
   ```

## Code Style

- We use [Black](https://github.com/psf/black) for code formatting
- We use [flake8](https://flake8.pycqa.org/) for linting
- We use type hints throughout the codebase
- We follow Google-style docstrings

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest src/tests/test_agents.py
```

## Code Quality Checks

Before submitting a PR, please run:

```bash
# Format code
black .

# Check formatting
black --check .

# Lint code
flake8 .

# Type checking (optional)
mypy src/
```

## Submitting Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Make your changes and add tests

3. Run the quality checks above

4. Commit your changes:
   ```bash
   git commit -m "Add amazing feature"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```

6. Create a Pull Request

## Pull Request Guidelines

- Include a clear description of what your change does
- Add tests for new functionality
- Ensure all existing tests pass
- Update documentation if needed
- Keep PRs focused and atomic

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features.

## Questions?

Feel free to open an issue for any questions about contributing!
