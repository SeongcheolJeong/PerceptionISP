# Development Guide

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Install `.[raw]`, `.[ml]`, or `.[camerae2e]` only for tests that require those
capabilities.

## Package Rules

- Sensor and ISP primitives belong in `core`.
- Dataset-specific I/O and split logic belongs in `datasets`.
- Parameter optimization belongs in `training`.
- Metric computation and claim decisions belong in `evaluation`.
- HTML/text presentation belongs in `reporting`.
- Use absolute `perception_isp.<group>...` imports across package boundaries.
- Optional dependencies must be imported at the point of use and produce a
  clear installation error.

## CLI Rules

Add a workflow module with `main(argv=None) -> int`, then register it in
`perception_isp.cli.COMMANDS`. Do not add another project-level console script.
The module parser remains responsible for its own detailed options.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m pytest
python -m compileall -q src tests examples
```

Unit tests must not require downloaded datasets. Resource-dependent tests
should skip with an explicit reason. Any refactor must keep the full baseline
suite passing before an experiment is resumed.

## Artifacts

Write generated content below the path selected by `PERCEPTION_ISP_OUTPUT`.
Do not commit:

- datasets or archives;
- model checkpoints;
- NPZ/NPY tensors;
- training runs and caches;
- generated HTML report trees.

Commit only small fixtures, summary tables needed by maintained docs, and code
that can regenerate the evidence.

## Reproducibility Metadata

Every claim-oriented run should record:

- source commit and command;
- dataset manifest/split hash;
- seed, model checkpoint, image size, epochs, batch, and optimizer;
- source/target CFA, remosaic fraction, LensPSF parameters;
- HumanISP and PerceptionISP settings;
- confidence/IoU thresholds and label mapping;
- sample-level outputs needed for bootstrap intervals and counterexamples.

## Before Push

```bash
git diff --check
git status --short
rg -n '/Users/|gho_|github_pat_' README.md docs src tests examples scripts
python -m unittest discover -s tests -p 'test_*.py'
```

Research archives may contain historical local paths; maintained source and
manuals must not.
