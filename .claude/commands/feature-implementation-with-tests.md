---
name: feature-implementation-with-tests
description: Workflow command scaffold for feature-implementation-with-tests in kultivait.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-implementation-with-tests

Use this workflow when working on **feature-implementation-with-tests** in `kultivait`.

## Goal

Implements a new feature or backend and adds corresponding unit tests.

## Common Files

- `src/kultivait/config.py`
- `tests/test_config.py`
- `src/kultivait/backends.py`
- `tests/test_backends.py`
- `src/kultivait/cli.py`
- `tests/test_llamacpp_survey.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or add implementation file in src/kultivait/
- Edit or add corresponding test file in tests/

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.