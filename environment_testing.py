"""
Skill: environmental-validity-testing

Prove benchmark claims match reality with negative controls.
Run oversubscription tests, memory limit verification, etc.
"""

import subprocess
import psutil
import os
import json


def test_thread_oversubscription(model_fn, n_threads_requested: int, n_cpus_available: int) -> dict:
    """
    Negative control: request more threads than CPU cores available.
    
    If sandbox is working correctly, should see no speedup (possibly slower from oversubscription).
    This proves cgroup/cpuset enforcement isn't leaking.
    
    Returns: {"sandbox_working": bool, "duration_x2_threads": float, "duration_x1_threads": float}
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
    print("Bug: x2 version faster → Constraints not enforced\n")
    
    return {
        "threads_x2_duration_s": round(duration_x2, 2),
        "threads_x1_duration_s": round(duration_x1, 2),
        "sandbox_working": sandbox_working,
        "conclusion": "PASS" if sandbox_working else "FAIL"
    }


def verify_memory_limits(memory_limit_mb: int) -> dict:
    """
    Verify that requesting more memory triggers OOM or gets capped.
    """
    import tracemalloc
    
    tracemalloc.start()
    
    # Allocate incrementally
    allocations = []
    max_allocation = None
    
    for i in range(100):
        try:
            chunk = [0] * (memory_limit_mb * 1024 * 1024 // 8)  # 8 bytes per item
            allocations.append(chunk)
            max_allocation = len(allocations)
        except MemoryError:
            break
        except Exception as e:
            print(f"Allocation failed at iteration {i}: {e}")
            break
    
    tracemalloc.stop()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.clear_traces()
    
    return {
        "peak_allocated_mb": round(peak / (1024 * 1024), 1),
        "limit_enforced": peak <= memory_limit_mb * 1.2,  # Allow 20% overhead
        "test_passed": peak <= memory_limit_mb * 1.1
    }


def get_observed_constraints() -> dict:
    """
    Print all constraints actually observed by this process.
    Should be printed before every benchmark for transparency.
    """
    constraints = {}
    
    # CPU affinity
    try:
        result = subprocess.run(["taskset", "-cp", str(os.getpid())], 
                              capture_output=True, text=True)
        constraints["cpu_affinity"] = result.stdout.strip().split(": ")[-1]
    except FileNotFoundError:
        constraints["cpu_affinity"] = "not available"
    
    # Cgroup limits
    cgroup_path = "/sys/fs/cgroup"
    if Path(cgroup_path).exists():
        try:
            with open(Path(cgroup_path) / "memory.max", "r") as f:
                val = f.read().strip()
                constraints["memory_max"] = val if val != "max" else "unlimited"
        except FileNotFoundError:
            constraints["memory_max"] = "cgroup v2 not available"
        
        try:
            with open(Path(cgroup_path) / "cpu.max", "r") as f:
                val = f.read().strip()
                constraints["cpu_quota"] = val
        except FileNotFoundError:
            constraints["cpu_quota"] = "not available"
    
    # Process-level info
    proc = psutil.Process()
    constraints["available_cpus"] = proc.cpu_count()
    constraints["current_threads"] = proc.num_threads()
    
    # Platform info
    constraints["platform"] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "cpu_model": get_cpu_model_name()
    }
    
    return constraints


def print_constraints_report(constraints: dict):
    """Print human-readable constraint report."""
    print("\n" + "="*60)
    print("OBSERVED CONSTRAINTS (proof of benchmark environment)")
    print("="*60)
    
    for key, value in constraints.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key.upper()}: {value}")
    
    print("="*60 + "\n")


def get_cpu_model_name() -> str:
    """Get CPU model name (platform-dependent)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except:
        pass
    return "unknown"


if __name__ == "__main__":
    # Run constraint verification before benchmarks
    print("Verifying benchmark environment...")
    constraints = get_observed_constraints()
    print_constraints_report(constraints)
    
    # Save JSON evidence
    with open("results/benchmark_evidence.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "constraints": constraints
        }, f, indent=2)
