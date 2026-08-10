# Metrics Pipeline Skills

Store all measurements as JSON source files; regenerate documentation programmatically from code — never hand-edit result tables.

## Skills in this Category

### generation
Generate Markdown tables and reports directly from JSON metrics sources.

**When to use:** All experimental reporting  
**Structure:**
```
results/
├── accuracy.json        # variant × format × split metrics
├── benchmarks.json      # latency/RSS/CPU% per configuration  
└── architecture.json    # params/GFLOPs/transfer rates
```

**Features:** Auto-regeneration, diff-friendly numeric records

## Example Usage

```python
from Skills.metrics_pipeline.generation import save_accuracy_results, generate_readme_table_from_json

# Save accuracy results to JSON
save_accuracy_results(results=[
    {"variant": "v2", "format": "INT8", "split": "test", 
     "mAP@0.5": 0.9494, "mAP@0.5:0.95": 0.6779},
    {"variant": "v2", "format": "TORCH", "split": "test",
     "mAP@0.5": 0.9539, "mAP@0.5:0.95": 0.6911}
])

# Regenerate README table
table = generate_readme_table_from_json("accuracy")
print(table)  # Output formatted Markdown table
```

## Philosophy

JSON is source of truth → Markdown is generated output.  
This enables reproducibility audits and makes it impossible to lie about numbers.
