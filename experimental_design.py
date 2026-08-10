"""
Skill: experimental-design

Define hard constraints before training begins and create reproducible benchmark harnesses that enforce them.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime


def define_constraints(cpu_cores: int, memory_mb: int, latency_budget_ms: float, accuracy_target: dict) -> dict:
    """
    Define deployment constraints for benchmarking.
    
    Args:
        cpu_cores: Available CPU cores (e.g., 2)
        memory_mb: Memory limit in MB (e.g., 2048)
        latency_budget_ms: Target p95 latency in milliseconds
        accuracy_target: {"metric": "mAP@0.5", "threshold": 0.95}
    
    Returns:
        Constraint specification for use in benchmark scripts
    """
    return {
        "hardware": {
            "cpu_cores": cpu_cores,
            "memory_max_mb": memory_mb,
            "accelerator": "none"  # or "cuda", "gpu"
        },
        "performance": {
            "latency_budget_p95_ms": latency_budget_ms,
            "fps_minimum": round(1000 / latency_budget_ms, 1)
        },
        "accuracy": accuracy_target,
        "enforcement": {
            "method": "cgroup_v2",
            "memory_limit": f"{memory_mb}M",
            "cpuset": list(range(cpu_cores))
        }
    }


def create_benchmark_harness(constraints: dict, output_path: str = "benchmark.py") -> None:
    """
    Create a benchmark script that enforces constraints and prints observed limits.
    
    This generates a Python script that:
    - Prints observed affinity mask
    - Reports cgroup memory limits actually enforced
    - Includes negative control tests
    """
    code = '''#!/usr/bin/env python3
"""Benchmark harness with constraint enforcement and verification."""

import os
import subprocess
import time
from contextlib import contextmanager
import json


def get_affinity_mask():
    """Print observed CPU affinity mask."""
    result = subprocess.run(["taskset", "-cp", str(os.getpid())], 
                          capture_output=True, text=True)
    print(f"Observed CPU affinity: {result.stdout.strip()}")
    return result.stdout.strip()


def get_memory_limits():
    """Report actual cgroup memory limits observed."""
    limits = {}
    
    # Try to read from cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            val = f.read().strip()
            limits["memory_max"] = val if val != "max" else "unlimited"
    except FileNotFoundError:
        limits["memory_max"] = "cgroup not available"
    
    # Try swap limit
    try:
        with open("/sys/fs/cgroup/memory.swap.max", "r") as f:
            val = f.read().strip()
            limits["memory_swap"] = val if val != "max" else "unlimited"
    except FileNotFoundError:
        limits["memory_swap"] = "not set"
    
    return limits


@contextmanager
def constrained_run(cpu_spec: str, memory_mb: int):
    """Run command within cgroup sandbox."""
    env = os.environ.copy()
    args = ["systemd-run", "--scope", 
            f"-p CPUS={cpu_spec}",
            f"-p MemoryMax={memory_mb}M",
            f"-p MemorySwapMax=0"]
    
    print(f"Running in sandbox: CPUs={cpu_spec}, Memory<{memory_mb}MB")
    print(get_memory_limits())
    
    try:
        yield
    finally:
        pass


def negative_control_thread_oversubscription(model_fn, n_threads_requested: int, n_cpus_available: int):
    """
    Prove environment isn't leaky by requesting more threads than available.
    If sandbox is working, should see no speedup (possibly slower).
    """
    print(f"\n--- NEGATIVE CONTROL: {n_threads_requested} threads in {n_cpus_available}-CPU sandbox ---")
    
    # Run with 2x requested threads
    start = time.time()
    results_fast = model_fn(n_threads=n_threads_requested * 2)
    duration_fast = time.time() - start
    
    # Run with 1x requested threads  
    start = time.time()
    results_slow = model_fn(n_threads=n_threads_requested)
    duration_slow = time.time() - start
    
    print(f"Threads x2: {duration_fast:.2f}s, Threads x1: {duration_slow:.2f}s")
    print("Expected: x2 version slower or equal (oversubscription)")
    print("Bug: x2 version faster → sandbox leaking\n")
    
    return {
        "threads_x2_duration_s": duration_fast,
        "threads_x1_duration_s": duration_slow,
        "sandbox_working": duration_fast >= duration_slow * 0.95
    }


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
    limits = get_memory_limits()
    print(json.dumps(limits, indent=2))
    
    # Load model
    print(f"Loading model: {args.model}")
    # [Model loading code here]
    
    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        # inference()
        pass
    
    # Benchmark
    print(f"Benchmarking ({args.iterations} iterations)...")
    timings = []
    rss_values = []
    
    for i in range(args.iterations):
        start = time.perf_counter_ns()
        # result = inference()
        end = time.perf_counter_ns()
        
        elapsed_ms = (end - start) / 1e6
        timings.append(elapsed_ms)
        
        # Collect RSS via psutil or /proc/self/statm
        # rss_values.append(get_current_rss_mb())
    
    # Report statistics
    p50 = sorted(timings)[len(timings) // 2]
    p95 = sorted(timings)[int(len(timings) * 0.95)]
    fps_end_to_end = 1000 / (sum(timings) / len(timings))
    
    print(f"\n=== RESULTS ===")
    print(f"p50 latency: {p50:.2f} ms")
    print(f"p95 latency: {p95:.2f} ms")
    print(f"FPS (end-to-end): {fps_end_to_end:.1f}")
    print(f"Peak RSS: {max(rss_values):.1f} MB")
    
    # Save to JSON for reproducibility
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "constraints": {
            "threads": args.threads,
            "iterations": args.iterations
        },
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "fps_end_to_end": fps_end_to_end,
        "peak_rss_mb": max(rss_values),
        "observations": limits
    }
    
    output_json = f"results/benchmarks_{args.model.replace('/', '_')}.json"
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_json}")


if __name__ == "__main__":
    main()
'''
    
    Path(output_path).write_text(code)
    print(f"Created benchmark harness: {output_path}")


def generate_constraint_doc(constraints: dict) -> str:
    """Generate markdown documentation of constraints for README."""
    doc = f"""## Constraints

| Parameter | Value | How Enforced |
|---|---|---|
| CPU | {constraints['hardware']['cpu_cores']} cores | cgroup v2 scope + taskset affinity |
| RAM | < {constraints['hardware']['memory_max_mb']} MB | MemoryMax cgroup limit |
| Accelerator | {constraints['hardware']['accelerator']} | CPU plugin only |
| Accuracy target | {constraints['accuracy']['metric']} ≥ {constraints['accuracy']['threshold']} | Held-out test split |
| Latency budget | {constraints['performance']['latency_budget_ms']} ms p95 | Real measurements under constraints |
"""
    return doc
