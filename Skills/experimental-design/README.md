# Experimental Design Skills

Design ML experiments under hard deployment constraints rather than idealized conditions.

## Skills in this Category

### constraints
Define deployment constraints (CPU cores, RAM, latency budget) before training begins.

**When to use:** Any edge/deployment-constrained ML project  
**Input:** Hardware specs, accuracy target, performance requirements  
**Output:** Constraint specification, benchmark harness setup

### benchmarking
Create reproducible benchmark harnesses that enforce constraints and include negative controls.

**When to use:** Publishing performance claims  
**Features:** Constraint verification, oversubscription tests, memory limit validation  
**Output:** Benchmark scripts with observed constraint reporting

## Example Usage

```python
from Skills.experimental_design.constraints import define_constraints
from Skills.experimental_design.benchmarking import create_benchmark_harness

# Step 1: Define constraints
constraints = define_constraints(
    cpu_cores=2,
    memory_mb=2048,
    latency_budget_ms=15.0,
    accuracy_target={"metric": "mAP@0.5", "threshold": 0.95}
)

# Step 2: Create benchmark harness
create_benchmark_harness(constraints, output_path="benchmark.py")
```

## Philosophy

Every claim must be backed by measurement under actual constraints, not benchmark ideals.
