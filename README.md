# Constraint-Driven ML Research Skills

A modular skill library for conducting rigorous, reproducible machine learning research under real deployment constraints. Based on the YOLOv11/OpenVINO face detection methodology: measuring everything, not estimating anything.

## Philosophy

**Every line answers:** "Would an agent/researcher miss this without help?"

If the answer is no, don't include it. If yes, make it actionable.

## Core Principles

1. **Measure under actual constraints** — Don't benchmark on your laptop; benchmark in the same cgroup/memory limits as production
2. **Trace where losses happen** — Separate export loss from quantisation loss from architecture changes
3. **Track weight transfer explicitly** — Parameter counts lie; pretrained tensor reuse doesn't
4. **Negative controls prove environment integrity** — Test that sandboxing actually works
5. **Generate metrics from code, not manually** — JSON source files + regeneration scripts = no hand-typed numbers

## Available Skills

### `experimental-design`
Define hard constraints before training begins (CPU cores, RAM, latency budget, accuracy target). Create reproducible benchmark harness that enforces them with cgroups/taskset or equivalent containerization.

**When to use:** Any edge/deployment-constrained ML project

**Input requirements:**
- Hardware spec (CPU cores, RAM limit, accelerator availability)
- Accuracy metric and gate (e.g., mAP@0.5 ≥ 0.95)
- Performance target (FPS, latency p95, memory peak)

**Output artifacts:**
- Benchmark script that prints observed constraints (affinity masks, memory limits)
- Negative control tests proving sandbox isn't leaky
- Requirements file with exact versions used

---

### `data-profiling-before-modification`
Profile data properties *before* making any architectural decisions. For object detection: measure box scale distributions per detection head's receptive band. For time series: measure frequency content, seasonality, missingness patterns.

**When to use:** Before removing heads, changing stride, modifying preprocessing

**Key question answered:** "Is dropping P5 head justified by data, or just dogma?"

**Output artifacts:**
- Distribution plots saved to `docs/`
- Quantitative table of which heads/components are actually needed
- Decision rationale linking data properties to architecture choices

---

### `weight-transfer-accounting`
When modifying architectures (width scaling, block removal, layer addition), track how many pretrained tensors transfer vs get reinitialised. This catches cases where parameter count optimization hurts more than it helps.

**When to use:** Any fine-tuning scenario with <10k unique training samples, width/architecture modifications

**Measurement method:** Intersection of new model's state_dict keys/shapes against pretrained checkpoint

**Output artifacts:**
- Transfer rate table (% tensors inherited vs total params)
- Warning if transfer drops below threshold (~80% recommended for small datasets)
- Recommendations for keeping architecture close to stock when weights matter

---

### `conversion-boundary-analysis`
Separate accuracy/speed losses by stage: PyTorch→ONNX/TFLite/INT8. Often the conversion itself costs more than quantisation.

**When to use:** Any model export workflow

**Standard measurement sequence:**
1. FP32 native (baseline)
2. Exported FP32/FP16 (conversion loss)
3. Quantised INT8 (quantisation impact)

**Output artifacts:**
- Accuracy delta table showing each stage's contribution
- Speedup factor per precision level
- Recommendation on whether quantisation worth the pipeline complexity

---

### `reproducible-metrics-pipeline`
Store all measurements as JSON source files. Generate README/table documents programmatically from these sources. Never hand-edit result tables.

**When to use:** All experimental reporting

**Structure:**
```
results/
├── accuracy.json        # variant × format × split metrics
├── benchmarks.json      # latency/RSS/CPU% per configuration
├── architecture.json    # params/GFLOPs/transfer rates
└── dataset_stats.json   # box/image distributions, augmentation stats
```

**Generation script:** Regenerates every table in documentation from JSON

**Output artifacts:**
- Source-of-truth JSON files
- Markdown report regenerated from code
- Diff-friendly numeric records (can see what changed between runs)

---

### `environmental-validity-testing`
Test that benchmark claims match reality. Run oversubscription tests (more threads than CPU cores available) to prove cgroup/cpuset enforcement works. Verify memory limits trigger OOM kills when exceeded.

**When to use:** Publishing performance claims, deploying to constrained hardware

**Negative control examples:**
- Requesting 8 threads inside 2-CPU cgroup → should show no speedup (oversubscription penalty)
- Allocating >MemoryMax → should OOM kill or cap at limit
- Accessing third CPU → should be throttled/killed

**Output artifacts:**
- Evidence rows in benchmark tables proving isolation
- Printouts of observed constraints (every benchmark outputs affinity mask, memory limits)
- Documentation of any limitations in enforcement (WSL2 quirks, kernel versions)

---

### `statistical-uncertainty-reporting`
For small test sets, quantify noise floor and confidence intervals. Report when differences are statistically indistinguishable. Don't claim wins within noise.

**When to use:** Any evaluation with <1k test samples, especially classification/detection metrics

**Calculation methods:**
- Bootstrap confidence intervals for mAP/F1/Accuracy
- Effect size estimates for metric deltas
- Minimum meaningful difference given sample size

**Output artifacts:**
- Uncertainty annotations on result tables (⚠️ vs ✅ gates)
- Text explaining statistical indistinguishability ("0.9494 vs 0.950 not distinguishable")
- Sample size justification or limitation statements

---

### `limitation-documentation`
Explicitly document structural constraints bounding generalisability. Distinguish between:
- **Limitations:** Cannot fix even with infinite compute/data (small test set, single-source domain)
- **Incomplete searches:** Could fix but didn't due to practical constraints (untrained variants)
- **Results:** What you found, good or bad (target missed by X points)

**Output artifacts:**
- Dedicated limitations section in documentation
- Clear labels on incomplete experiments
- Honest failure reporting (missed targets, null results)

---

## How to Use These Skills

### For Agents Working in New Repos

1. **Read highest-value sources first:** README, Makefile/package config, lockfiles, CI workflows
2. **Identify constraint pattern:** Is this edge deployment? Small dataset? Export pipeline?
3. **Apply relevant skills:** Use `weight-transfer-accounting` if modifying architectures, use `conversion-boundary-analysis` if exporting
4. **Generate artifacts:** Create JSON metrics, profile plots, limitation docs
5. **Verify environment:** Run negative controls before publishing claims

### Customizing for Your Framework

All skills work across frameworks. Only the tooling changes:

| Skill | PyTorch | TensorFlow | ONNX Runtime | JAX |
|---|---|---|---|---|
| Weight transfer | `state_dict()` overlap | `model.get_weights()` comparison | Same via TorchScript intermediate | NumPy array key matching |
| Export tracing | torch.onnx/export | tf.lite.TFLiteConverter | onnxruntime.InferenceSession | jax.jit + xla.compile |
| Quantisation | NNCF, bitsandbytes | TFLite converter | ONNX Runtime quantisation | flax.linen.quantize |
| Constraint harness | cgroups+taskset | Container run flags | Docker compose limits | Kubernetes resource specs |

---

## Credits

Based on methodology from: [face-detection-openvino-edge](https://github.com/rubythalib-ai/face-detection-openvino-edge)

Core insights:
- Architecture decisions driven by data profiles, not heuristics
- Weight transfer accounting reveals hidden costs of parameter reduction
- Conversion loss often exceeds quantisation loss
- Negative controls prove benchmark integrity
- Reproducible metrics from code, not manual editing
# skills
