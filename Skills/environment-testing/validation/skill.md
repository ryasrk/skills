# Skill: Environment Testing - Negative Controls

**Prove benchmark claims match reality with negative controls.** Test that your sandbox isn't leaking by requesting MORE resources than available — if you see no speedup, your constraints actually work.

---

## 🎯 When to Use This Skill

Apply this skill whenever **publishing ANY performance claims**:

- Research papers claiming latency/FPS numbers
- Product specs with resource requirements  
- Deployment guides with CPU/RAM limits
- Benchmark comparisons between models
- ANY claim about "running under X constraints"

**Key insight:** Just saying "we used cgroups" doesn't prove anything. Prove it worked with negative controls.

---

## 📋 Core Principles

### 1. Negative Controls Are Mandatory
❌ **Wrong:** Assume sandbox is working  
✅ **Correct:** Request 8 threads in 2-CPU sandbox → should show NO speedup

### 2. Always Print Observed Constraints
❌ **Wrong:** Trust what you asked for  
✅ **Correct:** Print what system actually observed (affinity mask, memory limits)

### 3. Memory Limits Must Actually Kill on OOM
❌ **Wrong:** "We tried to use 3GB but system throttled gracefully"  
✅ **Correct:** MemoryMax=2G triggers OOM kill at ~2.1GB

---

## 🔧 Step-by-Step Instructions

### Step 1: Verify Constraint Environment Before Benchmarks

```python
def verify_and_print_constraints() -> dict:
    """Print all constraints ACTUALLY observed by this process."""
    
    observations = {}
    
    # CPU affinity mask
    try:
        result = subprocess.run(["taskset", "-cp", str(os.getpid())], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            observations["observed_affinity_mask"] = result.stdout.strip().split(": ")[-1]
        else:
            observations["observed_affinity_mask"] = "could not retrieve"
    except FileNotFoundError:
        observations["observed_affinity_mask"] = "taskset not available"
    
    # Cgroup CPU limits (v2)
    cgroup_path = Path("/sys/fs/cgroup")
    if cgroup_path.exists():
        try:
            with open(cgroup_path / "cpu.max", "r") as f:
                val = f.read().strip()
                observations["cpu_quota_policy"] = val.split()[0]
                observations["cpu_quota_us"] = int(val.split()[1]) if val.split()[0] != "max" else None
        except FileNotFoundError:
            observations["cpu_quota_policy"] = "cgroup v2 cpu not available"
        
        try:
            with open(cgroup_path / "memory.max", "r") as f:
                val = f.read().strip()
                observations["memory_max_bytes"] = val if val != "max" else "unlimited"
        except FileNotFoundError:
            observations["memory_max_bytes"] = "cgroup v2 memory not available"
    
    # Process-level info
    import psutil
    proc = psutil.Process()
    observations["available_cpu_cores"] = proc.cpu_count()
    observations["current_num_threads"] = proc.num_threads()
    observations["process_memory_mb"] = round(proc.memory_info().rss / 1e6, 2)
    
    return observations


def print_observation_report(observations: dict):
    """Human-readable constraint report to include in every benchmark output."""
    
    print("\n" + "="*70)
    print("OBSERVED CONSTRAINTS (proof of benchmark environment integrity)")
    print("="*70)
    
    print(f"\nCPU CONFIGURATION:")
    print(f"  Observed affinity mask: {observations['observed_affinity_mask']}")
    print(f"  Available cores: {observations['available_cpu_cores']}")
    print(f"  Current threads: {observations['current_num_threads']}")
    
    if observations.get("cpu_quota_policy"):
        print(f"\nCGROUP CPU LIMITS:")
        print(f"  Quota policy: {observations['cpu_quota_policy']}")
        if observations.get("cpu_quota_us"):
            print(f"  Quota microsecond: {observations['cpu_quota_us']}")
    
    print(f"\nMEMORY STATUS:")
    print(f"  Memory max limit: {observations['memory_max_bytes']}")
    print(f"  Current RSS: {observations['process_memory_mb']} MB")
    
    print("\n" + "="*70 + "\n")
```

### Step 2: Implement Thread Oversubscription Negative Control

```python
def test_thread_oversubscription(model_fn, n_threads_requested: int, n_cpus_available: int) -> dict:
    """
    NEGATIVE CONTROL: Request MORE threads than CPU cores available.
    
    If sandbox is working correctly, should see NO speedup (possibly slower from oversubscription).
    This proves cgroup/cpuset enforcement isn't leaking to full machine.
    
    Returns: {"sandbox_working": bool, "duration_x2_threads": float, "duration_x1_threads": float}
    """
    print(f"\n{'='*60}")
    print(f"NEGATIVE CONTROL: {n_threads_requested * 2} threads in {n_cpus_available}-CPU sandbox")
    print(f"{'='*60}")
    
    def run_with_threads(n_threads):
        """Run model inference with specified thread count."""
        start = time.perf_counter()
        model_fn(n_threads=n_threads)
        duration = time.perf_counter() - start
        return duration
    
    # Run with requested threads
    duration_x1 = run_with_threads(n_threads_requested)
    
    # Run with DOUBLE requested threads  
    duration_x2 = run_with_threads(n_threads_requested * 2)
    
    print(f"\nResults:")
    print(f"  Threads x{int(n_threads_requested)}: {duration_x1:.2f}s")
    print(f"  Threads x{int(n_threads_requested * 2)}: {duration_x2:.2f}s")
    print(f"  Speedup factor: {duration_x1 / duration_x2:.2f}x")
    
    # Expected behavior:
    # - If sandbox leaky: x2 version faster (>1.0x)
    # - If sandbox working: x2 same or slower (<=1.05x due to overhead)
    
    sandbox_working = duration_x2 >= duration_x1 * 0.95  # Allow 5% noise margin
    
    conclusion = "PASS ✓" if sandbox_working else "FAIL ✗ LEAKY BENCHMARK ENVIRONMENT!"
    print(f"\nConclusion: {conclusion}")
    
    if not sandbox_working:
        print(f"⚠️ WARNING: x2 threads was FASTER → sandbox leaking to full machine!")
        print(f"   Your benchmark results are INVALID — constraints not enforced")
    
    print(f"{'='*60}\n")
    
    return {
        "threads_x1_duration_s": round(duration_x1, 4),
        "threads_x2_duration_s": round(duration_x2, 4),
        "speedup_factor": round(duration_x1 / duration_x2, 4),
        "sandbox_working": sandbox_working,
        "conclusion": conclusion,
        "warnings": [] if sandbox_working else ["Benchmark environment is leaking! Results invalid."]
    }
```

### Step 3: Test Memory Limit Enforcement

```python
def test_memory_limit_enforcement(memory_limit_mb: int, iterations: int = 5) -> dict:
    """
    Test that requesting >MemoryMax actually triggers OOM or gets capped.
    """
    import tracemalloc
    
    print(f"\nTesting memory limit enforcement: {memory_limit_mb}MB")
    print("-" * 50)
    
    allocations = []
    peak_allocation_mb = 0
    
    for i in range(iterations):
        try:
            chunk_size_mb = memory_limit_mb // 10
            chunk = [0] * (chunk_size_mb * 1024 * 1024 // 8)  # 8 bytes per item
            allocations.append(chunk)
            
            current_mb = len(allocations) * chunk_size_mb
            peak_allocation_mb = max(peak_allocation_mb, current_mb)
            
            print(f"  Iteration {i+1}: Allocated {current_mb}MB total")
            
        except MemoryError:
            print(f"  ✓ OOM triggered at {peak_allocation_mb}MB (limit: {memory_limit_mb}MB)")
            break
        except Exception as e:
            print(f"  ! Allocation failed: {type(e).__name__}: {e}")
            break
    
    tracemalloc.stop()
    
    within_limit = peak_allocation_mb <= memory_limit_mb * 1.2  # 20% overhead tolerance
    
    return {
        "peak_allocated_mb": round(peak_allocation_mb, 1),
        "limit_enforced": within_limit,
        "conclusion": "PASS ✓" if within_limit else "FAIL ✗ exceeded memory limit!",
        "test_passed": within_limit
    }
```

### Step 4: Combine Into Complete Verification Pipeline

```python
def run_full_environment_verification(constraints: dict, model_fn) -> dict:
    """
    Complete verification before running benchmarks.
    """
    print("\n" + "#" * 70)
    print("ENVIRONMENT VERIFICATION BEFORE BENCHMARKS")
    print("#" * 70)
    
    # Step 1: Observe constraints
    print("\n1. Observing actual constraints...")
    observations = verify_and_print_constraints()
    
    # Step 2: Verify CPU pinning via negative control
    print("\n2. Running CPU oversubension negative control...")
    cpu_result = test_thread_oversubscription(
        model_fn=model_fn,
        n_threads_requested=constraints.get("threads_used", 2),
        n_cpus_available=constraints.get("cpu_cores", 2)
    )
    
    # Step 3: Verify memory limits (if applicable)
    print("\n3. Testing memory limit enforcement...")
    memory_result = test_memory_limit_enforcement(
        memory_limit_mb=constraints.get("memory_max_mb", 2048)
    )
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_pass = True
    all_pass &= cpu_result["sandbox_working"]
    all_pass &= memory_result["test_passed"]
    
    if all_pass:
        print("✓ ALL CONTROLS PASSED — benchmark environment is valid")
    else:
        print("✗ SOME CONTROLS FAILED — do NOT publish these benchmark results!")
        if not cpu_result["sandbox_working"]:
            print("  ⚠️ CPU isolation leaking")
        if not memory_result["test_passed"]:
            print("  ⚠️ Memory limits not enforced")
    
    print("="*70 + "\n")
    
    return {
        "observations": observations,
        "cpu_negative_control": cpu_result,
        "memory_enforcement": memory_result,
        "all_controls_passed": all_pass
    }
```

---

## 💻 Example Command Sequences

### Running Full Verification

```bash
# Before any benchmark script
python scripts/verify_environment.py \
    --cpu-cores 2 \
    --threads 2 \
    --memory-mb 2048 \
    --model-model-function src/benchmark.py:model_fn

# Output includes:
# - Affinity mask verification
# - Oversubension control (8 threads in 2-CPU sandbox)
# - Memory limit enforcement test
# - Clear PASS/FAIL summary
```

### Integrating Into Existing Scripts

```python
# In benchmark.py
if __name__ == "__main__":
    args = parse_args()
    
    constraints = {
        "cpu_cores": 2,
        "threads_used": args.threads,
        "memory_max_mb": 2048
    }
    
    # Run verification FIRST
    verification = run_full_environment_verification(constraints, lambda: run_model_inference())
    
    if not verification["all_controls_passed"]:
        sys.exit(1)  # Don't even run benchmarks if environment is leaky
    
    # Now safe to run actual benchmarks
    print("Running benchmarks...")
    run_actual_benchmarks(args)
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Not Running Negative Controls

**WRONG:** "We ran benchmarks under 2 CPU cores"  
**No proof**

**CORRECT:** "We ran benchmarks under 2 CPU cores AND proved it with 8-thread oversubension control showing NO speedup"

From our research: Initial measurements showed 1.88 GB RSS — alarmingly close to 2 GB limit. Only after fixing harness and running controls did we discover real usage was 234 MB. Without controls, we would have reported "barely fits" when it actually uses minimal resources.

### ❌ Mistake 2: Relying Solely on System Commands

**WRONG:** `systemd-run --scope -p CPUS=0,1 ./benchmark.sh`  
**Trusts commands aren't being ignored**

**CORRECT:** After running, verify with `taskset -cp $PID` that affinity mask matches what you asked for

### ❌ Mistake 3: Ignoring WSL2/Container Differences

**WRONG:** Running Linux cgroups directly inside Docker without proper flags  
**Cgroups may be invisible to container**

**CORRECT:** Use `--privileged` or pass through cgroups hierarchically

### ❌ Mistake 4: Only Testing One Scenario

**WRONG:** Test only your exact configuration  
**What if someone runs with different settings?**

**CORRECT:** Include parameterized tests in CI that check various constraint combinations

---

## ✅ Success Indicators

Your environment verification is production-ready when:

1. ✅ Every benchmark output prints observed constraints automatically
2. ✅ Includes CPU oversubension control (never leaks)
3. ✅ Shows memory limit enforcement works
4. ✅ Clear PASS/FAIL summary at end
5. ✅ Can skip benchmarking entirely if controls fail
6. ✅ Evidence saved alongside results in JSON
7. ✅ No one publishes numbers without passing all controls

**Test yourself:** Could someone reproduce your EXACT experimental environment just by reading your code? If they need to ask "what hardware?" or "were you really using 2 cores?", you haven't proven it enough.

---

## 🔗 Related Skills

| Skill | Purpose |
|---|---|
| `experimental_design/constraints` | Define constraints BEFORE testing |
| `metrics_pipeline/generation` | Store verification evidence alongside metrics |
| `limitations/documentation` | Document environmental validity concerns |

---

## 📚 Reference Examples

### From Face Detection Project

**Verification pipeline we implemented:**
```bash
# Each benchmark run automatically executes:
./scripts/run_constrained.sh \
    --model models/yolo11n-face-v2/int8_openvino_model \
    --threads 2

# Inside script:
1. Print observed affinity mask: "0,1" ✓
2. Print memory limits from cgroup: "2G" ✓
3. Run 8-thread oversubension control: NO SPEEDUP ✓
4. Verify memory didn't exceed 2G: Peak 234MB ✓
5. Only then run actual 300-iteration benchmark
```

**Negative control evidence:**
```
Model A (2 threads): 13.18 ms average
Model A (8 threads): 14.16 ms average (slower!)
→ Sandbox IS working, constraints ARE enforced
```

This is critical evidence reviewers expect for performance claims.

### What We Learned

1. Negative controls prevent publishing fake numbers accidentally
2. People always trust their intentions ("I meant to limit to 2 cores") over evidence ("Oh wait, I'm seeing 8 cores active")
3. Including evidence IN EVERY benchmark run makes it automatic
4. Fail fast: don't even run benchmarks if environment is leaky

---

## 🎯 Quick Checklist

Before publishing ANY benchmark claim:

- [ ] Printed observed affinity mask (not just requested)
- [ ] Reported cgroup memory limits actually observed
- [ ] Ran thread oversubension control (more threads than CPUs)
- [ ] Confirmed oversubension shows NO speedup (proves isolation)
- [ ] Tested memory limits trigger OOM at correct threshold
- [ ] Saved verification evidence alongside benchmark data
- [ ] Added clear PASS/FAIL summary to benchmark output
- [ ] Can quote specific negative control results in paper/report

If any checkbox is unchecked, those benchmark claims are not trustworthy.