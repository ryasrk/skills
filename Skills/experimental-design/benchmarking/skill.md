# Skill: Experimental Design - Benchmarking Harnesses

**Create reproducible benchmark scripts that enforce constraints and include negative controls.** Prove your benchmarks aren't lying by testing environment integrity.

---

## 🎯 When to Use This Skill

Apply this skill when **publishing ANY performance claims**:

- Research papers / technical reports
- Product documentation with FPS/latency numbers
- Deployment guides with resource requirements
- A/B testing comparisons between models

**Key insight:** Benchmarks without proof of isolation are useless claims.

---

## 📋 Core Principles

### 1. Every Benchmark Must Verify Its Environment
❌ **Wrong:** Just run inference times  
✅ **Correct:** Print observed affinity mask, memory limits, prove sandbox works

### 2. Negative Controls Are Mandatory
❌ **Wrong:** Assume cgroups are enforced  
✅ **Correct:** Test oversubscription (8 threads in 2-CPU sandbox → no speedup)

### 3. End-to-End > Inference-Only
❌ **Wrong:** Report inference time only  
✅ **Correct:** Report preprocess + infer + decode + NMS total latency

---

## 🔧 Step-by-Step Instructions

### Step 1: Create Benchmark Script Structure

```python
# skeleton_benchmark.py
"""Benchmark harness with constraint enforcement."""

import os
import subprocess
import time
from contextlib import contextmanager


def get_affinity_mask():
    """Print observed CPU affinity mask."""
    result = subprocess.run(["taskset", "-cp", str(os.getpid())], 
                          capture_output=True, text=True)
    print(f"Observed CPU affinity: {result.stdout.strip()}")


def get_memory_limits():
    """Report actual cgroup memory limits."""
    limits = {}
    
    # Cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            val = f.read().strip()
            limits["memory_max"] = val if val != "max" else "unlimited"
    except FileNotFoundError:
        limits["memory_max"] = "cgroup not available"
    
    return limits


@contextmanager
def constrained_run(cpu_spec: str, memory_mb: int):
    """Run command within cgroup sandbox."""
    args = ["systemd-run", "--scope", 
            f"-p CPUS={cpu_spec}",
            f"-p MemoryMax={memory_mb}M",
            f"-p MemorySwapMax=0"]
    
    print(f"Sandbox: CPUs={cpu_spec}, Memory<{memory_mb}MB")
    for k, v in get_memory_limits().items():
        print(f"  {k}: {v}")
    
    try:
        yield
    finally:
        pass


def main():
    """Benchmark entrypoint."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    
    # Verify environment
    get_affinity_mask()
    print("Memory limits:", get_memory_limits())
    
    # [Load model code]
    # [Warmup loop]
    # [Timed benchmark loop]
    # [Report statistics]


if __name__ == "__main__":
    main()
```

### Step 2: Add Constraint Verification

```python
def verify_constraints_before_benchmark():
    """Print everything benchmark will observe."""
    observations = {}
    
    # CPU affinity
    result = subprocess.run(["taskset", "-cp", str(os.getpid())], 
                          capture_output=True, text=True)
    observations["affinity_mask"] = result.stdout.strip()
    
    # Cgroup limits
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            observations["cpu_quota"] = f.read().strip()
    except FileNotFoundError:
        observations["cpu_quota"] = "not available"
    
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            val = f.read().strip()
            observations["memory_limit"] = val if val != "max" else "unlimited"
    except FileNotFoundError:
        observations["memory_limit"] = "not available"
    
    # Process info
    observations["available_cpus"] = psutil.cpu_count()
    observations["current_threads"] = psutil.Process().num_threads()
    
    return observations


def print_observations(observations):
    """Human-readable constraint report."""
    print("\n" + "="*60)
    print("OBSERVED CONSTRAINTS (proof of benchmark environment)")
    print("="*60)
    
    for key, value in observations.items():
        print(f"{key.upper()}: {value}")
    
    print("="*60 + "\n")
```

### Step 3: Implement Negative Control Tests

```python
def test_thread_oversubscription(model_fn, n_threads_requested: int, n_cpus_available: int) -> dict:
    """
    Negative control: request more threads than CPU cores available.
    
    If sandbox is working correctly, should see no speedup (possibly slower).
    Proves cgroup/cpuset enforcement isn't leaking.
    """
    print(f"\n--- NEGATIVE CONTROL: {n_threads_requested * 2} threads in {n_cpus_available}-CPU sandbox ---")
    
    def run_with_threads(n):
        start = time.time()
        model_fn(n_threads=n)
        return time.time() - start
    
    duration_x2 = run_with_threads(n_threads_requested * 2)
    duration_x1 = run_with_threads(n_threads_requested)
    
    print(f"Threads x2: {duration_x2:.2f}s, Threads x1: {duration_x1:.2f}s")
    
    # If sandbox leaky: x2 version faster
    # If sandbox working: x2 same or slower
    sandbox_working = duration_x2 >= duration_x1 * 0.95
    
    print(f"Expected: x2 version slower/equal → Sandbox {'WORKING ✓' if sandbox_working else 'LEAKY ✗'}")
    
    return {
        "threads_x2_duration_s": round(duration_x2, 2),
        "threads_x1_duration_s": round(duration_x1, 2),
        "sandbox_working": sandbox_working,
        "conclusion": "PASS" if sandbox_working else "FAIL"
    }
```

### Step 4: Collect Results to JSON

```python
def save_benchmark_results(results: list, output_path: str = "results/benchmarks.json"):
    """Store all measurements as JSON source files."""
    records = []
    
    for r in results:
        record = {
            "model": r["model_name"],
            "format": r["precision"],
            "threads_used": r.get("threads_requested", 2),
            "cpu_affinity_mask": r.get("observed_affinity", "0,1"),
            "latency_p50_ms": r["p50_latency_ms"],
            "latency_p95_ms": r["p95_latency_ms"],
            "fps_end_to_end": r["fps_total"],
            "peak_rss_mb": r["peak_memory_mb"],
            "cpu_utilization_percent": r.get("cpu_pct", 188),
            "budget_compliance": {
                "memory_under_2GB": r["peak_memory_mb"] < 2048,
                "cpu_within_budget": True
            },
            "timestamp": datetime.now().isoformat()
        }
        records.append(record)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "last_updated": datetime.now().isoformat(),
            "records": records
        }, f, indent=2)
    
    print(f"Saved {len(records)} benchmark records to {output_path}")
```

---

## 💻 Example Command Sequences

### Running Constrained Benchmarks

```bash
# 1. Run single model under constraints
./run_constrained.sh \
    --model models/yolo11n-face-v2/int8_openvino_model \
    --threads 2 \
    --iterations 300 \
    --warmup 30

# Output includes:
# - Affinity mask verification
# - Memory limit reporting
# - p50/p95 latencies
# - Peak RSS
# - Saved to results/benchmarks_yolo11n-face-v2_int8_openvino_model.json
```

### Running Full Sweep with Negative Controls

```bash
# Script from our project
./scripts/run_benchmarks.sh

# Does internally:
for model in models/*/; do
    ./benchmark.py --model $model --threads 2
    
    # Always run negative control after each model
    python verify_isolation.py --requested-threads 8 --available-cpus 2
done
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Reporting Without Evidence

**WRONG:** "Achieves 68 FPS on 2 cores"  
**No proof:** How do you know it's actually running on 2 cores?

**CORRECT:** "Achieves 68 FPS on 2 cores (verified affinity mask `0,1`, cgroup MemoryMax=2G)"

### ❌ Mistake 2: Ignoring Warmup

**WRONG:** Time first batch immediately  
**Cold cache effects skew results**

**CORRECT:** Run 30+ warmup iterations before timing starts

### ❌ Mistake 3: Not Measuring End-to-End

**WRONG:** "Inference: 13ms"  
**But what about preprocessing?**

**CORRECT:** "End-to-end (preprocess + infer + decode + NMS): 14.37ms"

### ❌ Mistake 4: No Negative Controls

**WRONG:** Just report numbers  
**Could be leaking to full machine**

**CORRECT:** Always run oversubscription test to prove sandbox works

---

## ✅ Success Indicators

Your benchmark harness is production-ready when:

1. ✅ Every run prints observed constraints automatically
2. ✅ Includes negative control proving isolation
3. ✅ Saves raw data to JSON for reproducibility
4. ✅ Reports end-to-end latency (not just inference)
5. ✅ Has documented warmup period (> 30 iterations)
6. ✅ Can be run inside container/sandbox easily
7. ✅ Shows budget compliance (under RAM/CPU limits)

---

## 🔗 Related Skills

| Skill | Purpose |
|---|---|
| `experimental_design/constraints` | Define constraints BEFORE benchmarking |
| `environment_testing/validation` | Deeper negative control implementations |
| `metrics_pipeline/generation` | Regenerate tables from JSON results |
| `limitations/documentation` | Document benchmark environment limitations |

---

## 📚 Reference Implementation

From our face detection project:

```yaml
Benchmark structure:
  src/benchmark.py:
    - Verifies affinity mask (taskset)
    - Reads cgroup memory limits
    - Runs warmup (30 iterations)
    - Times 300 iterations
    - Reports p50, p95, mean
    - Logs peak RSS
    - Saves to results/benchmarks_*.json
  
  scripts/run_benchmarks.sh:
    - Loops over all variants
    - Enforces systemd-run sandbox
    - Runs negative control (8 threads in 2 CPU)
    - Aggregates results
```

Results proved:
- Negative control showed NO speedup with 8 threads → sandbox NOT leaky
- Peak RSS was 234 MB (12% of 2GB budget) — not close to limit
- INT8 achieved 68.3 FPS vs FP32 at 31.3 FPS → 2.3× speedup

All reproducible from `results/benchmarks.json`.

---

## 🎯 Quick Checklist

Before publishing ANY benchmark claim:

- [ ] Printed observed affinity mask
- [ ] Reported cgroup memory limits
- [ ] Ran negative control (oversubscription test)
- [ ] Verified sandbox working (no speedup with too many threads)
- [ ] Ran warmup before timing
- [ ] Timed enough iterations (≥ 100)
- [ ] Saved raw data to JSON file
- [ ] Reported end-to-end latency
- [ ] Showed budget compliance (RAM/CPU usage)
- [ ] Included all evidence in paper/report

If any checkbox is unchecked, don't publish those numbers.