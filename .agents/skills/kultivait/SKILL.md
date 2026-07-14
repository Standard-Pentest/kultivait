```markdown
# kultivait Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill covers the core development patterns, coding conventions, and workflows for contributing to the `kultivait` Python repository. It outlines how to implement new features with test coverage, update documentation, and adhere to the project's code style and commit message conventions.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for all Python files.  
  _Example:_  
  ```
  src/kultivait/backends.py
  tests/test_config.py
  ```

- **Import Style:**  
  Use relative imports within the package.  
  _Example:_  
  ```python
  from .config import load_config
  from .backends import Backend
  ```

- **Export Style:**  
  Use named exports (explicitly define what is exported).  
  _Example:_  
  ```python
  __all__ = ["Backend", "load_config"]
  ```

- **Commit Messages:**  
  Follow [Conventional Commits](https://www.conventionalcommits.org/) with prefixes such as `feat` and `docs`.  
  _Example:_  
  ```
  feat: add support for custom backend configuration
  docs: update README with backend usage instructions
  ```

## Workflows

### Feature Implementation with Tests
**Trigger:** When adding a new feature or backend with test coverage  
**Command:** `/new-feature-with-tests`

1. Edit or add the implementation file in `src/kultivait/`.
2. Edit or add the corresponding test file in `tests/`.
3. Ensure your code follows the coding conventions above.
4. Write a commit message using the `feat:` prefix.
5. Run all tests to verify your changes.

_Example:_
```python
# src/kultivait/backends.py
class NewBackend:
    def process(self, data):
        return data[::-1]
```

```python
# tests/test_backends.py
from kultivait.backends import NewBackend

def test_new_backend_process():
    backend = NewBackend()
    assert backend.process("abc") == "cba"
```

### Documentation Update
**Trigger:** When documenting new functionality or setup instructions  
**Command:** `/update-docs`

1. Edit or add markdown files in the documentation (e.g., `README.md`, `landing/start.md`).
2. Clearly describe new features, usage patterns, or setup steps.
3. Use the `docs:` prefix in your commit message.

_Example:_
```markdown
## New Backend Usage

To use the new backend, import and initialize it as follows:
```python
from kultivait.backends import NewBackend
backend = NewBackend()
```
```

## Testing Patterns

- **Test File Naming:**  
  Test files are named using the pattern `test_*.py` and are placed in the `tests/` directory.
  _Example:_  
  ```
  tests/test_config.py
  tests/test_backends.py
  ```

- **Test Framework:**  
  The specific test framework is not detected, but standard Python test conventions apply (e.g., using `pytest` or `unittest`).

- **Test Structure:**  
  Each test function should start with `test_` and assert expected behavior.
  _Example:_  
  ```python
  def test_config_loading():
      config = load_config("test.yaml")
      assert config["key"] == "value"
  ```

## Commands

| Command                   | Purpose                                                   |
|---------------------------|-----------------------------------------------------------|
| /new-feature-with-tests   | Scaffold a new feature or backend with corresponding tests|
| /update-docs              | Update or add documentation for new features or setups    |
```
