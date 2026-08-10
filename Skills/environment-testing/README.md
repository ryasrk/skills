# Environment Testing Skills

Prove benchmark claims match reality with negative controls and constraint verification.

## Skills in this Category

### validation
Run negative control tests to prove sandbox isn't leaking constraints.

**When to use:** Publishing performance claims  
**Negative controls:**
- Thread oversubscription: request 8 threads in 2-CPU sandbox → should see no speedup
- Memory limits: requesting >MemoryMax → should trigger OOM or get capped
- CPU pinning: verify taskset affinity mask matches cgroup configuration

**Output:** Evidence that benchmarks ran under actual constraints

## Example Usage

```python
from Skills.environment_testing.validation import test_thread_oversubscription

# Prove benchmark environment integrity
negative_control = test_thread_oversubscription(
    model_fn=my_model_inference,
    n_threads_requested=2,
    n_cpus_available=2
)

print(f"Sandbox working: {negative_control['sandbox_working']}")
if not negative_control['sandbox_working']:
    raise ValueError("Benchmark environment is leaky — results invalid!")
```

## Research Insight

Initial measurements showed ~1.88 GB RSS usage (close to 2 GB limit).  
After running negative controls and fixing benchmark harness design: real peak was **234 MB** (12% of budget).

Without negative controls, you would have reported "model barely fits" when it actually uses minimal resources.
