# Installation

## Core Install

The package is installable from a local checkout:

```bash
pip install .
```

For the tagged GitHub release:

```bash
pip install "variopt @ git+https://github.com/isty2e/variopt.git@v0.1.0"
```

The core install includes:

- `numpy`
- `scipy`
- `joblib`
- `typing_extensions`

## Optional Extras

Install extras only when you need those integrations.

```bash
pip install ".[test]"
pip install ".[docs]"
pip install ".[mpi]"
```

For a GitHub tag install, use the direct URL form with extras, for example:

```bash
pip install "variopt[docs] @ git+https://github.com/isty2e/variopt.git@v0.1.0"
```

Use them as follows:

- `test`: pytest-based test runner and local verification
- `docs`: MkDocs site build
- `mpi`: MPI-backed evaluators


## Verify The Install

The smallest sanity check is:

```python
from variopt import Problem, RealSpace, Study
from variopt.algorithms.population import CSAOptimizer
from variopt.evaluators import SequentialEvaluator
```

If those imports succeed, the base install is intact.
