# How Agents Should Use These Skills

## When Starting Work on a New ML Repository

### 1. First Investigation (Read-Only)
```python
# Identify constraint pattern
is_edge_deployment = any([
    "Makefile" in files,
    "cgroup" in readme_content,
    "RAM limit" in readme_content or "memory" in readme_content
])

# Check if research methodology exists
has_metrics_pipeline = Path("results/accuracy.json").exists()
has_benchmark_harness = Path("benchmark.py").exists()
```

**Decision tree:**
- If **yes** to both → repository uses constraint-driven methodology; read AGENTS.md first
- If **no** → consider applying these skills to improve rigor

### 2. Applying Skills Incrementally

#### For Training Projects

1. **Before architecture changes**: Run `profile_box_sizes()` or equivalent data profiling
   - Output: "Which detection heads are needed?"
   - Don't drop P5 head unless <5% of boxes fall in that range

2. **After modifying architectures**: Run `check_weight_transfer()` 
   - Input: variant config, pretrained checkpoint path
   - Warning if transfer_rate < 80% for datasets < 10k images

3. **During evaluation**: Set up `analyze_conversion_pipeline()`
   - Measure FP32-native → ONNX → INT8 separately
   - Report where accuracy losses actually occur

#### For Benchmarking Projects

1. **Before publishing results**: Run negative controls
   - Oversubscription test: request 8 threads in 2-CPU sandbox
   - Expect no speedup (or slowdown) proving environment integrity

2. **Always print observed constraints** before benchmarks:
   - CPU affinity mask (`taskset -cp`)
   - Cgroup memory limits (`cat /sys/fs/cgroup/memory.max`)
   - Include this evidence in every benchmark result

3. **Store all metrics as JSON**, regenerate tables from code
   - Never hand-edit numbers in README
   - Diff-friendly numeric records enable reproducibility audits

#### For Limitations Section

Use `suggest_limitations_from_experiment()` to automatically identify:
- Small test sets → statistical uncertainty
- Untrained variants → search space gaps  
- Single dataset source → domain transfer concerns
- Virtualized environments → environmental validity warnings

Then write them honestly rather than hiding disappointing results.

## Common Mistakes Agents Make

### Mistake 1: Dropping Heads Based on Dogma
```python
# ❌ WRONG: Following standard YOLO practice
model = remove_head(model, "P5")

# ✅ CORRECT: After data profiling shows it's safe
box_profile = profile_box_sizes(dataset_path)
if box_profile["P5_stride_32"]["share_percent"] > 5:
    raise ValueError("DO NOT DROP P5 HEAD - only 52% of boxes would remain detectable")
```

### Mistake 2: Assuming Width Scaling Is Free
```python
# ❌ WRONG: Reducing width and assuming same performance gain
model_v1 = scale_width(model, 0.1875)  # Seems smaller!

# ✅ CORRECT: Check weight transfer first
transfer = check_weight_transfer("v1.yaml", "coco.pt")
print(f"{transfer['transfer_rate']:.1f}% weights inherited - high risk!")
# Result: V1 has 40% fewer params but 1.6 mAP worse due to random initialization
```

### Mistake 3: Reporting Conversion Loss as Quantisation Loss
```python
# ❌ WRONG: Comparing FP32→INT8 only
native_mAP = evaluate_model(fp32=True)
quantised_mAP = evaluate_model(int8=True)
report_loss(native_mAP - quantised_mAP, "quantisation")

# ✅ CORRECT: Trace intermediate stages
fp32_ov = export_to_openvino(fp32=True)
ov_fp32_loss = native_mAP - fp32_ov.mAP  # -0.0055 pt conversion loss
int8_ov = export_to_openvino(int8=True)  
int8_loss = fp32_ov.mAP - int8_ov.mAP   # +0.0009 pt (essentially free!)
```

### Mistique 4: Trusting Benchmarks Without Negative Controls
```bash
# ❌ WRONG: Running benchmarks without proving isolation
./run_benchmarks.sh  # Could leak to full machine

# ✅ CORRECT: Prove sandbox isn't leaky
systemd-run --scope -p MemoryMax=2G ./run_benchmarks.sh &
pid=$!
sleep 2
taskset -cp $pid  # Verify affinity mask observed correctly
```

## Quick Reference Cheat Sheet

| Goal | Skill to Apply | Key Question Answered |
|---|---|---|
| Deciding architecture changes | `data_profiling.profile_box_sizes()` | Which heads/components actually needed? |
| Modifying width/architecture | `weight_transfer.check_weight_transfer()` | Will I lose more than I save? |
| Exporting model | `conversion_analysis.analyze_conversion_pipeline()` | Where does accuracy actually go? |
| Publishing benchmarks | `environment_testing.test_thread_oversubscription()` | Am I sure constraints weren't leaked? |
| Writing limitations section | `limitations_docs.suggest_limitations_from_experiment()` | What bounds my conclusions? |
| Measuring under budget | `experimental_design.define_constraints()` | What am I optimizing against actual deployment? |

## Example Agent Workflow

```python
# 1. Read highest-value sources first
readme = Path("README.md").read_text()
makefile = Path("Makefile").read_text()

# 2. Identify if constraint-driven methodology already exists
if "mAP@0.5 ≥ 0.95" in readme and "2 cores" in readme:
    # This repo is constraint-driven; look for existing skills
    if Path("AGENTS.md").exists():
        agentic_guidance = Path("AGENTS.md").read_text()
    
# 3. Before modifying architectures:
box_profile = profile_box_sizes("data/train/")
recommendation = recommend_head_removal(box_profile)
if recommendation["_overall"]["can_remove_any_head"]:
    print(f"SAFE TO MODIFY: {recommendation}")
else:
    raise ValueError(recommendation["_overall"]["evidence"])

# 4. After training variants:
for variant in ["v0", "v1", "v2"]:
    transfer = check_weight_transfer(f"configs/models/{variant}.yaml", "pretrained/coco.pt")
    if transfer["warning_threshold_exceeded"]:
        print(f"⚠️ {variant} inherits only {transfer['transfer_rate']:.1f}% weights")
        
# 5. During eval:
pipeline = MetricsPipeline()
for split in ["train", "val", "test"]:
    results = evaluate_accuracy(variant="v2", format="INT8", split=split)
    pipeline.add_accuracy_record(...)

# 6. Before publishing:
limitations = suggest_limitations_from_experiment({
    "test_set_size": len(test_images),
    "configured_variants": configs.keys(),
    "trained_variants": trained_variants,
    "dataset_source": dataset_name,
})
write_limitations_section(limitations, "README.md")
```

This ensures every claim is backed by measurement, not estimation.
