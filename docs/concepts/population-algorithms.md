# Population Algorithms

The population family currently includes:

- CSA
- Differential Evolution
- Genetic Algorithm
- Clearing GA
- Restricted Tournament GA
- Species Conserving GA

## Practical Guidance

- use DE as the broad continuous baseline inside the native library
- use GA variants when population diversity policy is the main knob
- use CSA when you want the richer structured-space defaults and policy surface

The canonical import surface is:

```python
from variopt.algorithms.population import (
    ClearingGeneticAlgorithmOptimizer,
    CSAOptimizer,
    DifferentialEvolutionOptimizer,
    GeneticAlgorithmOptimizer,
    RestrictedTournamentGeneticAlgorithmOptimizer,
    SpeciesConservingGeneticAlgorithmOptimizer,
)
```

GA-family variants share the same run-method boundary as
`GeneticAlgorithmOptimizer` but expose different diversity policies:

- `ClearingGeneticAlgorithmOptimizer` clears nearby winners so one niche does
  not dominate the whole population.
- `RestrictedTournamentGeneticAlgorithmOptimizer` replaces only among nearby
  candidates under a supplied diversity metric.
- `SpeciesConservingGeneticAlgorithmOptimizer` preserves representatives from
  discovered species under a supplied diversity metric.
