# Coding Standards

These guidelines keep the *Efficient RNN Project* consistent and easy to extend while we explore pruning strategies.

## General Principles
- **Clarity first**: favor readable code over clever shortcuts; explain the *why* when behavior is non-obvious.
- **Determinism**: expose random seeds and note when stochasticity is unavoidable (e.g. Monte-Carlo pruning).
- **Incremental change**: touching a module means leaving it cleaner—add docstrings, type hints, or small refactors if they aid understanding.

## Python Style
- Target **Python 3.8+** compatibility; use `typing` annotations on public functions, returning `None` explicitly when useful.
- Follow **PEP 8** spacing/naming except where math notation is clearer (e.g. short loop indices like `t`, `i`, `j`).
- Keep line length ≤ 100 characters; wrap with parentheses instead of backslashes.
- Prefer **dataclasses** for configuration blobs and named tuples for lightweight structured returns.
- Use f-strings for formatting and `pathlib.Path` for filesystem work outside performance-critical loops.

## Documentation
- Each public module (`ctrnn_training/*`) must have a header docstring with its purpose (add one when editing the file).
- Public classes and functions require docstrings describing:
  - purpose/behavior,
  - inputs and expected shapes or types,
  - return values or side effects.
- Inline comments are reserved for non-obvious logic or math; avoid narrating self-explanatory statements.
- When adding new pruning strategies, include a short rationale in the docstring referencing the biological or algorithmic motivation.

## Testing and Experiments
- Wrap runnable scripts with `if __name__ == "__main__":` guards; ensure CLI entry points round-trip through `argparse`.
- Provide deterministic smoke tests or command examples in PR/commit descriptions, especially for new pruning modes.
- Record experiment parameters and metrics via `append_results_csv`; new metrics should be added as columns with clear names.

## Dependencies
- Prefer standard library + PyTorch/NumPy. Gate optional imports (`neurogym`, etc.) and fail gracefully with clear messages.
- Do not modify environment-wide seeds inside libraries—set them in the CLI or experiment drivers.

## Git / Workflow Notes
- Each commit should encapsulate one logical change (e.g., “Add STDP pruning strategy”) and reference any relevant experiments.
- Avoid force-pushing shared branches without coordination; for sweeping changes, create a design doc or issue first.

Following these practices keeps the research codebase reproducible and approachable for collaborators. Update this document when conventions evolve.
