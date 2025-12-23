# Similarity Scoring Method Comparison Report

This report compares four similarity scoring methods based on four diversity metrics.

## Methods Compared

- **LLM-based (Answer2)** (answer2)
- **Embedding-based** (embedding)
- **Feature-based** (feature)
- **Hybrid** (hybrid)

## Metrics

- **Parameter Coverage (PC)** (PC)
- **Behavior Coverage (PEC)** (PEC)
- **Trajectory Diversity (TCD)** (TCD)
- **Behavior Matrix (BCM)** (BCM)

## Results Summary

| Method | PC (mean±std) | PEC (mean±std) | TCD (mean±std) | BCM (mean±std) | Runs |
|--------|---------------|----------------|----------------|----------------|------|
| LLM-based (Answer2) | 0.167±0.000 | 0.725±0.000 | 0.668±0.000 | 0.243±0.000 | 1 |
| Embedding-based | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0 |
| Feature-based | 0.125±0.000 | 0.633±0.000 | 0.855±0.000 | 0.191±0.000 | 1 |
| Hybrid | 0.174±0.000 | 0.633±0.000 | 0.693±0.000 | 0.224±0.000 | 1 |

## Detailed Statistics

### LLM-based (Answer2)

**Parameter Coverage (PC)**:
- Mean: 0.1669
- Std: 0.0000
- Min: 0.1669
- Max: 0.1669
- Runs: 1

**Behavior Coverage (PEC)**:
- Mean: 0.7245
- Std: 0.0000
- Min: 0.7245
- Max: 0.7245
- Runs: 1

**Trajectory Diversity (TCD)**:
- Mean: 0.6682
- Std: 0.0000
- Min: 0.6682
- Max: 0.6682
- Runs: 1

**Behavior Matrix (BCM)**:
- Mean: 0.2429
- Std: 0.0000
- Min: 0.2429
- Max: 0.2429
- Runs: 1

### Embedding-based

**Parameter Coverage (PC)**:
- Mean: 0.0000
- Std: 0.0000
- Min: 0.0000
- Max: 0.0000
- Runs: 0

**Behavior Coverage (PEC)**:
- Mean: 0.0000
- Std: 0.0000
- Min: 0.0000
- Max: 0.0000
- Runs: 0

**Trajectory Diversity (TCD)**:
- Mean: 0.0000
- Std: 0.0000
- Min: 0.0000
- Max: 0.0000
- Runs: 0

**Behavior Matrix (BCM)**:
- Mean: 0.0000
- Std: 0.0000
- Min: 0.0000
- Max: 0.0000
- Runs: 0

### Feature-based

**Parameter Coverage (PC)**:
- Mean: 0.1249
- Std: 0.0000
- Min: 0.1249
- Max: 0.1249
- Runs: 1

**Behavior Coverage (PEC)**:
- Mean: 0.6334
- Std: 0.0000
- Min: 0.6334
- Max: 0.6334
- Runs: 1

**Trajectory Diversity (TCD)**:
- Mean: 0.8552
- Std: 0.0000
- Min: 0.8552
- Max: 0.8552
- Runs: 1

**Behavior Matrix (BCM)**:
- Mean: 0.1913
- Std: 0.0000
- Min: 0.1913
- Max: 0.1913
- Runs: 1

### Hybrid

**Parameter Coverage (PC)**:
- Mean: 0.1737
- Std: 0.0000
- Min: 0.1737
- Max: 0.1737
- Runs: 1

**Behavior Coverage (PEC)**:
- Mean: 0.6327
- Std: 0.0000
- Min: 0.6327
- Max: 0.6327
- Runs: 1

**Trajectory Diversity (TCD)**:
- Mean: 0.6934
- Std: 0.0000
- Min: 0.6934
- Max: 0.6934
- Runs: 1

**Behavior Matrix (BCM)**:
- Mean: 0.2243
- Std: 0.0000
- Min: 0.2243
- Max: 0.2243
- Runs: 1

