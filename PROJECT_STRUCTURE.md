# Project Structure Overview

## 📁 Directory Hierarchy

```
skills-repo/
├── README.md                    # Main documentation
├── USAGE.md                     # Agent workflow guide
├── setup.py                     # Package installer
│
├── Skills/                      # Core research skills (7 categories)
│   ├── experimental-design/     # Constraint definition & benchmarking
│   │   ├── constraints/         # Hardware/software limits
│   │   └── benchmarking/        # Reproducible test harnesses
│   │
│   ├── data-profiling/          # Data analysis before modifications
│   │   └── analysis/            # Distribution profiling tools
│   │
│   ├── weight-transfer/         # Pretrained parameter tracking
│   │   └── accounting/          # Weight transfer metrics
│   │
│   ├── conversion-analysis/     # Export pipeline loss tracing
│   │   └── boundary/            # Conversion vs quantisation
│   │
│   ├── metrics-pipeline/        # Reproducible metrics storage
│   │   └── generation/          # JSON → Markdown generators
│   │
│   ├── environment-testing/     # Benchmark integrity verification
│   │   └── validation/          # Negative control implementations
│   │
│   └── limitations/             # Honest documentation of constraints
│       └── documentation/       # Auto-limitation generators
│
├── Plugins/                     # Framework-specific implementations (3)
│   └── frameworks/
│       ├── pytorch/             # PyTorch/TorchScript plugins
│       ├── tensorflow/          # TensorFlow/TFLite plugins
│       └── onnx/                # ONNX Runtime plugins
│
└── Tools/                       # Utilities and helpers (3)
    ├── utils/                   # Common utility functions
    ├── generators/              # Code/report generators
    └── parsers/                 # Configuration/result file readers
```

## 🔑 Key Design Principles

### 1. Category > Sub-category Structure
Each skill belongs to a **Category** (e.g., `weight-transfer`) with a **Sub-category** (e.g., `accounting`).

**Why:** Clear mental model for organizing related functionality while maintaining granularity.

### 2. Each Skill Has Three Components
```
Skills/category-name/subcategory/
├── README.md              # Documentation, examples, philosophy
├── __init__.py           # Python package initialization
└── skill.py              # Main implementation
```

**Benefits:**
- Self-documenting
- Importable as Python modules
- Easy to extend with multiple subcategories

### 3. Plugins Are Framework-Specific
`Plugins/frameworks/{pytorch,tensorflow,onnx}/` contain equivalent implementations using framework-native APIs.

**Why:** Same methodology, different tooling per framework.

### 4. Tools Support Multiple Skills
`Tools/` contains shared utilities that can be imported by any skill or plugin.

**Examples:** Constraint enforcement, metric calculators, file parsers.

## 🧠 Mental Model for Using This Structure

### Finding the Right Skill

1. **What are you trying to do?**
   - Define constraints? → `experimental-design/`
   - Profile data first? → `data-profiling/`
   - Track weight inheritance? → `weight-transfer/`
   - Analyze export losses? → `conversion-analysis/`
   - Store metrics repro? → `metrics-pipeline/`
   - Prove benchmarks real? → `environment-testing/`
   - Write honest limitations? → `limitations/`

2. **Which sub-category?**
   - Most skills have only one sub-category right now
   - As we add more features, they'll branch into new sub-categories

### Extending the Structure

#### Adding a New Skill Category
```bash
mkdir -p Skills/new-category/subcategory/
touch Skills/new-category/__init__.py
touch Skills/new-category/subcategory/__init__.py
touch Skills/new-category/subcategory/skill.py
```

#### Adding Framework Plugin
```bash
mkdir -p Plugins/frameworks/jax/
touch Plugins/frameworks/jax/__init__.py
touch Plugins/frameworks/jax/skill.py
```

#### Adding Shared Tool
```bash
touch Tools/utils/shared_function.py
```

## 📊 Statistics

| Component | Count | Description |
|---|---|---|
| **Skill Categories** | 7 | Core research methodologies |
| **Skill Sub-categories** | 7 | Specific implementations |
| **Plugins** | 3 | Framework support (PyTorch, TF, ONNX) |
| **Tools Directories** | 3 | Utils, Generators, Parsers |
| **Python Packages** | 20+ | All directories properly organized |
| **Documentation Files** | 8 | READMEs for each skill category |

## 🚀 Future Extensions

Potential additions to this structure:

### More Skill Categories
- `model-selection/` - Architecture search strategies
- `augmentation-strategies/` - Data augmentation design
- `evaluation-methods/` - Metric definitions beyond mAP/F1

### Additional Plugins
- `jax/` - JAX/XLA-specific implementations
- `triton/` - Triton compiler optimizations
- `openvino/` - OpenVINO export plugins

### Expanded Tools
- `generators/templates/` - Jinja2 templates for reports
- `parsers/yaml_configs/` - YAML configuration parsing
- `utils/visualization/` - Plotting and charting utilities

## 🎯 How Agents Should Navigate

When starting work on a new ML project:

1. **Read Skills/experimental-design/** first to define constraints
2. **Read Skills/data-profiling/** before modifying architectures  
3. **Use Plugins/frameworks/{your-framework}/** for framework-specific implementations
4. **Call Tools/** utilities when implementing common operations
5. **Reference READMEs** in each category for usage examples

The structure mirrors the research workflow itself:
Design → Profile → Implement → Verify → Document → Publish

## 🌐 Repository URL

https://github.com/ryasrk/skills.git
