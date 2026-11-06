# Contributing

Bug reports and pull requests are welcome through GitHub issues and pull requests.

## Set up

```bash
uv sync
uvx pre-commit install
```

## Before a commit

```bash
uvx pre-commit run --all-files   # format, lint, types, spelling, size limits
uv run pytest                    # tests; `-m "not slow and not gpu"` is what CI runs
```

GPU tests run on a machine with a GPU and never in CI.

## What a change needs

- One logical change per commit, with a subject such as `fix(solver): keep the step size positive`.
- A bug fix adds a test that fails without the fix.
- A change in behaviour updates `docs/` in the same commit.
- Do not weaken a test to make it pass. A changed tolerance, skip or xfail needs a written reason.

General practice follows the [Scientific Python Development Guide](https://learn.scientific-python.org/development/).
