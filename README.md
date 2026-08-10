# Constraint-Driven ML Research Skills Library

A modular skill library for conducting rigorous, reproducible machine learning research under real deployment constraints. Based on the YOLOv11/OpenVINO face detection methodology.

## 📚 Project Structure

```
skills-repo/
├── README.md                 # This file
├── USAGE.md                  # How agents should use these skills
├── setup.py                  # Package installation
│
├── Skills/                   # Core research skills
│   ├── experimental-design/  # Define constraints & benchmark harnesses
│   │   ├── constraints/      # Hardware constraint definitions
│   │   └── benchmarking/     # Reproducible benchmark scripts
│   │
│   ├── data-profiling/       # Profile data before architecture changes
│   │   └── analysis/         # Distribution profiling tools
│   │
│   ├── weight-transfer/      # Track pretrained parameter inheritance
│   │   └── accounting/       # Weight transfer measurement
│   │
│   ├── conversion-analysis/  # Separate export losses by stage
│   │   └── boundary/         # Conversion vs quantisation tracing
│   │
│   ├── metrics-pipeline/     # Reproducible metrics storage
│   │   └── generation/       # JSON → Markdown generation
│   │
│   ├── environment-testing/  # Prove benchmark integrity
│   │   └── validation/       # Negative control tests
│   │
│   └── limitations/          # Honest limitation documentation
│       └── documentation/    # Auto-generate limitations sections
│
├── Plugins/                  # Framework-specific implementations
│   └── frameworks/
│       ├── pytorch/          # PyTorch/TorchScript plugins
│       ├── tensorflow/       # TensorFlow/TFLite plugins
│       └── onnx/             # ONNX Runtime plugins
│
└── Tools/                    # Utilities and generators
    ├── utils/                # Common utilities
    ├── generators/           # Code/report generators
    └── parsers/              # Config/result file readers
```

## 🔥 Core Philosophy

**Every line answers:** "Would an agent/researcher miss this without help?"

If yes → Include it as actionable guidance.  
If no → Leave it out (it's obvious).

### Key Principles

1. **Measure under actual constraints** — Don't benchmark on your laptop; benchmark in the same cgroup/memory limits as production
2. **Trace where losses happen** — Separate export loss from quantisation loss from architecture changes
3. **Track weight transfer explicitly** — Parameter counts lie; pretrained tensor reuse doesn't
4. **Negative controls prove environment integrity** — Test that sandboxing actually works
5. **Generate metrics from code, not manually** — JSON source files + regeneration scripts = no hand-typed numbers

## 🛠️ Available Skills

| Category | Skill | When to Use |
|---|---|---|
| `experimental-design` | Define hard constraints before training | Any edge/deployment-constrained project |
| `data-profiling` | Profile data before modifications | Before removing heads/components |
| `weight-transfer` | Track pretrained parameter inheritance | Modifying architectures with <10k samples |
| `conversion-analysis` | Trace accuracy losses by stage | Any model export workflow |
| `metrics-pipeline` | Store metrics as JSON source | All experimental reporting |
| `environment-testing` | Prove benchmarks aren't lying | Publishing performance claims |
| `limitations` | Document structural constraints | Before presenting/publishing results |

## 💡 Example Workflow

```python
# 1. Design experiment under constraints
from Skills.experimental_design.constraints import define_constraints

constraints = define_constraints(
    cpu_cores=2, memory_mb=2048,
    accuracy_target={"mAP@0.5": 0.95}
)

# 2. Profile data before modifying architecture
from Skills.data_profiling.analysis import profile_box_sizes

box_profile = profile_box_sizes("data/train/")

# 3. Check weight transfer after modifications
from Skills.weight_transfer.accounting import check_weight_transfer

transfer = check_weight_transfer("model_v1.yaml", "coco.pt")
print(f"{transfer['transfer_rate_percent']}% weights transferred")

# 4. Analyze conversion pipeline
from Skills.conversion_analysis.boundary import analyze_conversion_pipeline

results = analyze_conversion_pipeline("trained_model", "test_data/")

# 5. Save metrics reproducibly
from Skills.metrics_pipeline.generation import save_accuracy_results

save_accuracy_results(results)

# 6. Verify benchmark environment
from Skills.environment_testing.validation import test_thread_oversubscription

negative_control = test_thread_oversubscription(...)

# 7. Generate honest limitations
from Skills.limitations.documentation import suggest_limitations_from_experiment

limitations = suggest_limitations_from_experiment(experiment_metadata)
```

See [USAGE.md](USAGE.md) for detailed agent workflows.

## 🎯 From Real Research

This methodology comes from our [YOLOv11 face detection project](https://github.com/rubythalib-ai/face-detection-openvino-edge):

- **Architecture decisions driven by data profiles**, not heuristics
- **Weight transfer accounting reveals hidden costs** of parameter reduction
- **Conversion loss often exceeds quantisation loss**
- **Negative controls prove benchmark integrity**
- **Reproducible metrics from code, not manual editing**

## 🌍 Framework Agnostic

All skills work across frameworks:
- ✅ PyTorch (Ultralytics, custom models)
- ✅ TensorFlow/Keras
- ✅ ONNX Runtime
- ✅ JAX (via conversion plugins)

The tooling changes but the methodology stays identical.

## 📖 License

MIT — same as the parent project.
