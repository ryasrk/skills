# Skill: Experimental Design - Constraints Definition

**Define hard deployment constraints BEFORE training begins.** Don't optimize against vague targets — specify exact CPU cores, memory limits, latency budgets, and accuracy gates.

---

## 🎯 When to Use This Skill

Apply this skill when starting **ANY edge/deployment-constrained ML project**:

- Mobile deployment (< 2GB RAM)
- Embedded systems (limited CPU cores)  
- Real-time inference (latency budget < 50ms)
- Cost-sensitive deployments (cloud GPU hours)
- Energy-constrained devices (battery-powered)

**Key insight:** Optimization without constraints is meaningless. Define your budget first.

---

## 📋 Core Principles

### 1. Hard Constraints Over Soft Targets
❌ **Wrong:** "Fast enough for production"  
✅ **Correct:** "p95 latency ≤ 15ms on 2 CPU cores"

### 2. Measure Under Actual Conditions
❌ **Wrong:** Benchmarks on full-power machine  
✅ **Correct:** Run benchmarks inside cgroup sandbox matching production

### 3. Accuracy Gates Are Binary Gates
❌ **Wrong:** "High mAP is better"  
✅ **Correct:** "mAP@0.5 ≥ 0.95 or reject model"

---

## 🔧 Step-by-Step Instructions

### Step 1: Define Hardware Budget

```json
{
    "hardware": {
        "cpu_cores": 2,           // Max cores available
        "memory_max_mb": 2048,    // Memory cap (RAM + swap)
        "accelerator": "none",    // or "cuda", "gpu"
        "cache_size_mb": null     // If applicable
    }
}
```

**What to measure:**
- Count available CPU cores (`nproc --all`)
- Check RAM limit (`free -h` or `cat /proc/meminfo`)
- Identify accelerator availability (`lspci | grep -i vga`)

### Step 2: Define Performance Budget

```json
{
    "performance": {
        "latency_budget_p95_ms": 15.0,     // Maximum acceptable p95 latency
        "fps_minimum": 60,                 // Minimum throughput
        "energy_budget_joules_per_inference": null  // If battery-powered
    }
}
```

**What to ask stakeholders:**
- What's the maximum acceptable response time?
- How many requests per second must we handle?
- Are there thermal/battery constraints?

### Step 3: Define Accuracy Gates

```json
{
    "accuracy_target": {
        "metric": "mAP@0.5",                // Or F1, PER, ROUGE, etc.
        "threshold": 0.95,                  // HARD gate
        "split": "test",                    // Held-out validation set
        "secondary_metrics": {              // For monitoring only
            "precision": 0.90,
            "recall": 0.85
        }
    }
}
```

**Critical:** Choose metrics that align with business impact, not convenience.

### Step 4: Document Enforcement Method

```yaml
enforcement:
  method: cgroup_v2              # Linux containers
  memory_limit: "2G"             # In bytes with unit suffix
  cpuset: [0, 1]                 # Which CPU cores
  affinity_mask: taskset         # Tool for pinning
  sandbox_command: |
    systemd-run --scope \
      -p CPUS=0,1 \
      -p MemoryMax=2G \
      ./benchmark_script.sh
```

**Alternatives:** Docker containers, Kubernetes resource specs, WSL2 limits

---

## 💻 Example Command Sequences

### For YOLO/Detection Projects

```bash
# 1. Create constraint spec file
cat > constraints.json << 'EOF'
{
    "cpu_cores": 2,
    "memory_mb": 2048,
    "latency_budget_ms": 15.0,
    "accuracy_target": {
        "metric": "mAP@0.5",
        "threshold": 0.95,
        "split": "test"
    }
}
EOF

# 2. Generate benchmark harness
python scripts/generate_benchmark_harness.py \
    --config constraints.json \
    --output benchmark.py

# 3. Run constrained benchmarks
./scripts/run_constrained.sh \
    --model models/yolo11n-face-v2/int8_openvino_model \
    --threads 2 \
    --iterations 300 \
    --warmup 30
```

### For General PyTorch Models

```bash
# Run evaluation under constraints
systemd-run --scope \
    -p CPUS=0,1 \
    -p MemoryMax=2G \
    python src/evaluate.py \
        --model checkpoint.pt \
        --data test_split.json
```

### For TensorFlow/TFLite

```bash
# Constrained TFLite export & bench
systemd-run --scope \
    -p MemoryMax=1G \
    python export_and_bench.py \
        --model keras_model.h5 \
        --tflite_output model.tflite \
        --backend tflite_micro
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Defining Constraints After Training

**WRONG:** Train freely, then ask "Can it fit?"
```python
model = train_large_model()  # No constraints applied
# Later: Oops, doesn't fit in target RAM!
```

**CORRECT:** Define constraints FIRST
```python
constraints = define_constraints(cpu=2, ram=2048)
# Then design model WITHIN those constraints
```

### ❌ Mistake 2: Vague Performance Targets

**WRONG:** "Optimize for speed"  
**Ambiguous:** Faster than what? On which hardware?

**CORRECT:** "Achieve 68 FPS end-to-end on 2 CPU cores"

### ❌ Mistake 3: Ignoring End-to-End Latency

**WRONG:** Benchmark only inference_time
```python
results = benchmark_inference(model)
```

**CORRECT:** Benchmark full pipeline including preprocessing/postprocessing
```python
results = benchmark_end_to_end(model)
```

### ❌ Mistake 4: Using Idealized Benchmarks

**WRONG:** Run on full machine, ignore constraints
```bash
python benchmark.py --model best_model.pt
```

**CORRECT:** Prove sandbox isn't leaky
```bash
systemd-run --scope -p MemoryMax=2G ./run_negative_control.sh
```

---

## ✅ Success Indicators

You've defined good constraints when:

1. ✅ Every researcher knows the budget before writing code
2. ✅ Benchmarks print observed constraints automatically
3. ✅ Negative controls prove isolation works correctly
4. ✅ Accuracy gates are explicit (pass/fail criteria)
5. ✅ All measurements are reproducible from constraint docs

**Test yourself:** Can someone else reproduce your exact environment just by reading your constraint docs?

---

## 🔗 Related Skills

| Skill | Purpose |
|---|---|
| `environment_testing/validation` | Verify sandbox isn't leaking constraints |
| `metrics_pipeline/generation` | Store results under these constraints |
| `conversion_analysis/boundary` | Measure losses within constraint budget |
| `limitations/documentation` | Document if constraints couldn't be met |

---

## 📚 Reference Examples

From our face detection project:

```json
{
    "hardware": {"cpu_cores": 2, "memory_max_mb": 2048},
    "performance": {"latency_budget_p95_ms": 15.0, "fps_minimum": 60},
    "accuracy": {"metric": "mAP@0.5", "threshold": 0.95},
    "enforcement": {
        "method": "cgroup_v2",
        "commands": [
            "systemd-run --scope -p MemoryMax=2G",
            "taskset -cp 0,1"
        ]
    }
}
```

Result: Deployed model achieves 68 FPS, 234MB RAM, mAP@0.5 = 0.9494 (missed by 0.0006).

---

## 🎯 Quick Checklist

Before starting any experiment:

- [ ] Defined CPU core count explicitly
- [ ] Defined memory limit (RAM + swap)
- [ ] Set latency/performance budget with units
- [ ] Chose accuracy metric AND threshold gate
- [ ] Documented enforcement method (cgroup/Docker/K8s)
- [ ] Created benchmark harness that enforces constraints
- [ ] Planned negative control tests
- [ ] Committed constraint docs before first training run

If any checkbox is unchecked, go back and define it.
