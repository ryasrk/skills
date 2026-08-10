# Skill: Metrics Pipeline - Reproducible Storage

**Store all measurements as JSON source files; regenerate documentation programmatically from code — never hand-edit result tables.**

---

## 🎯 When to Use This Skill

Apply this skill for **ALL experimental reporting**:

- Training runs (accuracy curves per epoch)
- Benchmark results (FPS, latency, RAM usage)  
- Architecture comparisons (params, FLOPs, transfer rates)
- Dataset statistics (distribution, augmentation effects)
- Any numeric claims you want others to reproduce

**Key insight:** Hand-typed numbers in READMEs are lies waiting to happen. JSON sources + regeneration scripts = truth.

---

## 📋 Core Principles

### 1. JSON is Source of Truth
❌ **Wrong:** Type numbers into Markdown manually  
✅ **Correct:** Save raw data to JSON, generate Markdown from it

### 2. Diff-Friendly Records Enable Audits
❌ **Wrong:** Single large table with no structure  
✅ **Correct:** Each record has timestamp, version, constraints observed

### 3. Regenerate Everything From Code
❌ **Wrong:** Copy-paste table from output to README  
✅ **Correct:** Script regenerates every table from JSON

---

## 🔧 Step-by-Step Instructions

### Step 1: Define Result Schema

```python
# Standardized schemas for different result types

accuracy_schema = {
    "last_updated": datetime.now().isoformat(),
    "records": [
        {
            "variant": str,           # Model variant name
            "format": str,            # TORCH, FP32, FP16, INT8, etc.
            "split": str,             # train, val, test
            "mAP@0.5": float,         # Main accuracy metric
            "mAP@0.5:0.95": float,    # Secondary metric
            "precision": float,       # For classification/detection
            "recall": float,          # For classification/detection
            "f1_score": float,        # Optional
            "threshold_used": float,  # Detection threshold
        }
    ]
}

benchmarks_schema = {
    "last_updated": datetime.now().isoformat(),
    "records": [
        {
            "model": str,
            "format": str,
            "threads_used": int,
            "cpu_affinity_mask": str,   # e.g., "0,1"
            "latency_p50_ms": float,
            "latency_p95_ms": float,
            "fps_end_to_end": float,
            "peak_rss_mb": float,
            "cpu_utilization_percent": float,
            "budget_compliance": {
                "memory_under_limit": bool,
                "cpu_within_budget": bool
            },
            "timestamp": str
        }
    ]
}

architecture_schema = {
    "last_updated": datetime.now().isoformat(),
    "variants": [
        {
            "name": str,
            "total_params": int,
            "total_gflops_640": float,
            "transfer_rate_percent": float,
            "pretrained_inheritance": int,  # transferred / total keys
            "category_breakdown": dict
        }
    ]
}
```

### Step 2: Implement Saving Functions

```python
def save_accuracy_results(results: list, output_path: str = "results/accuracy.json"):
    """Save accuracy metrics across variants × formats × splits."""
    
    records = []
    for r in results:
        record = dict(r)
        records.append(record)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "records": records
        }, f, indent=2)
    
    print(f"Saved {len(records)} records to {output_path}")


def save_benchmark_results(results: list, output_path: str = "results/benchmarks.json"):
    """Save latency/RAM/CPU% benchmark data per configuration."""
    
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
                "memory_under_limit": r["peak_memory_mb"] < 2048,
                "cpu_within_budget": True  # enforced by cgroup
            }
        }
        records.append(record)
    
    with open(output_path, "w") as f:
        json.dump({
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "records": records
        }, f, indent=2)
    
    print(f"Saved {len(records)} benchmark records to {output_path}")
```

### Step 3: Create Generation Script

```python
def generate_readme_table_from_json(metrics_type: str, output_section: str) -> str:
    """Regenerate Markdown table from JSON source file."""
    
    if metrics_type == "accuracy":
        data = load_accuracy_json()
        lines = ["| Variant | Format | Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |"]
        lines.append("|---|---|---|---|---|---|---|")
        
        # Sort by variant → format → split
        sorted_records = sorted(data["records"], key=lambda x: (x["variant"], x["format"], x["split"]))
        
        for r in sorted_records:
            gate_mark = " ✅" if r["mAP@0.5"] >= 0.95 else " ❌"
            line = f"| `{r['variant']}` | {r['format']} | {r['split']} | {r['mAP@0.5']:.4f}{gate_mark} | {r['mAP@0.5:0.95']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
            lines.append(line)
            
    elif metrics_type == "benchmarks":
        data = load_benchmark_json()
        lines = ["| Model | CPUs | Infer p50 | End-to-end p50 | FPS (e2e) | Peak RSS | Budget |"]
        lines.append("|---|---|---|---|---|---|---|")
        
        for r in sorted(data["records"], key=lambda x: x["model"]):
            budget_mark = "✅" if r["budget_compliance"]["memory_under_limit"] else "❌"
            fps_line = f"{r['fps_end_to_end']:.1f}"
            line = f"| `{r['model']}/{r['format']}` | {r['cpu_affinity_mask']} | {r['latency_p50_ms']:.2f} ms | {r['latency_p50_ms']*r['fps_end_to_end']/1000:.2f} ms | {fps_line} | {r['peak_rss_mb']:.1f} MB | {budget_mark} |"
            lines.append(line)
    
    return "\n".join(lines)


def regenerate_all_tables():
    """Generate all README tables from JSON sources."""
    
    print("\n=== ACCURACY TABLE ===\n")
    accuracy_table = generate_readme_table_from_json("accuracy", "# Accuracy Results")
    print(accuracy_table)
    
    print("\n=== BENCHMARK TABLE ===\n")
    benchmark_table = generate_readme_table_from_json("benchmarks", "# Benchmarks Under Constraints")
    print(benchmark_table)
    
    print("\n=== ARCHITECTURE TABLE ===\n")
    architecture_table = generate_architecture_table_from_json()
    print(architecture_table)
```

### Step 4: Implement Loader Functions

```python
def load_accuracy_json(path: str = "results/accuracy.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_benchmark_json(path: str = "results/benchmarks.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_architecture_json(path: str = "results/architecture.json") -> dict:
    with open(path) as f:
        return json.load(f)
```

---

## 💻 Example Command Sequences

### Adding Individual Records

```bash
# After each experiment, append record directly to JSON
python scripts/add_accuracy_record.py \
    --variant yolo11n-face-v2 \
    --format INT8 \
    --split test \
    --map50 0.9494 \
    --map5095 0.6779 \
    --precision 0.9262 \
    --recall 0.8938
```

**Automated after training script:**
```python
# In train.py at end of epoch
save_accuracy_record(
    variant=args.variant,
    format="TORCH",
    split="val",
    map50=val_map50,
    ...
)
```

### Regenerating Documentation

```bash
# Before committing any changes, regenerate tables
python scripts/regenerate_report.py

# Output shows exactly what will change in README
# If manual edits differ, they'll be overwritten by regeneration
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Hand-Typing Numbers

**WRONG:** Copy numbers from terminal output into README manually  
**Errors creep in, hard to verify, impossible to audit**

**CORRECT:** Always save to JSON first, regenerate from there

From our project: Initial reports had wrong V1 mAP values until we traced back to JSON source. Now everything auto-regenerated.

### ❌ Mistake 2: No Schema Versioning

**WRONG:** Change JSON structure mid-project  
**Breaks reproducibility**

**CORRECT:** Include `schema_version` in header, don't break backward compatibility

### ❌ Mistake 3: Missing Evidence Fields

**WRONG:** Just report numbers without context  
**Who knows under what conditions?**

**CORRECT:** Every record includes:
- `cpu_affinity_mask`: Which cores actually used?
- `budget_compliance`: Did we stay within limits?
- `timestamp`: When was this measured?
- `constraints_observed`: What environment was this running in?

### ❌ Mistake 4: One Giant File Per Experiment

**WRONG:** `experiment_20240810_final_v3_REAL.json`  
**Unstructured mess**

**CORRECT:** Separate concerns:
```
results/
├── accuracy.json      # Accuracy metrics only
├── benchmarks.json    # Performance metrics only
├── architecture.json  # Architecture comparison only
└── dataset_stats.json # Dataset properties only
```

---

## ✅ Success Indicators

Your metrics pipeline is production-ready when:

1. ✅ Every number in README can be traced to specific JSON record
2. ✅ Running `regenerate_report.py` updates ALL tables correctly
3. ✅ JSON diff between versions shows exactly what changed
4. ✅ New experiment automatically adds record to existing JSON
5. ✅ Can reconstruct entire paper from JSON files alone
6. ✅ Anyone can rerun generation script and get identical output
7. ✅ No human ever touches numbers after initial measurement

**Test yourself:** Could someone recreate your entire results section just by reading your JSON files and running your generation script? If yes, you've got it right.

---

## 🔗 Related Skills

| Skill | Purpose |
|---|---|
| `environment_testing/validation` | Prove environment integrity before storing results |
| `limitations/documentation` | Document incomplete searches in same schema |
| `experimental_design/constraints` | Store constraint details alongside metrics |

---

## 📚 Reference Examples

### From Face Detection Project

**Directory structure:**
```
results/
├── accuracy.json              # 96 records: 4 variants × 3 formats × 4 splits
├── benchmarks.json            # 10 records: 9 models + 1 negative control
├── architecture.json          # 4 variants with transfer analysis
├── exports.json               # Model export sizes per format
├── dataset_stats.json         # Box distributions, augmentation stats
├── architectures.json         # Params/GFLOPs breakdown
└── tables.md                  # Generated markdown tables (DON'T EDIT MANUALLY)
```

**Generation workflow:**
```bash
# Run full pipeline
make eval           # Generates accuracy.json
make bench          # Generates benchmarks.json
make report         # Regenerates all tables in README
```

**Result:** All tables in README are exact reflection of JSON — zero hand-typed numbers possible.

### What We Learned

1. Never trust your memory — always store raw data
2. JSON is far easier to diff than Markdown tables
3. Automation prevents accidental corruption of important numbers
4. Reviewers can verify your work by checking JSON against reported tables

---

## 🎯 Quick Checklist

For EVERY experiment result:

- [ ] Saved raw data to JSON immediately after measurement
- [ ] Included all constraint evidence in record (affinity mask, memory limits)
- [ ] Used consistent schema with version tag
- [ ] Separated concerns (accuracy vs performance vs architecture)
- [ ] Documented generation script that can recreate all reports
- [ ] Verified regenerated tables match published numbers exactly
- [ ] Committed both JSON and generation script to repository

If any checkbox is unchecked, you're not done with that experiment yet.