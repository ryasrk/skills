# Limitations Documentation Skills

Explicitly document structural constraints that bound how far results generalise — distinguish limitations from incomplete searches.

## Skills in this Category

### documentation
Automatically identify and write honest limitations sections based on experiment metadata.

**When to use:** Before publishing or presenting research  
**Types of limitations documented:**
- Statistical uncertainty (small test sets)
- Search space gaps (untrained variants)
- Domain transfer limits (single-source datasets)
- Environmental validity concerns (virtualized environments)
- Hardware capability dependencies (feature requirements)

**Output:** Markdown limitations section ready to insert into README

## Example Usage

```python
from Skills.limitations.documentation import suggest_limitations_from_experiment

# Auto-generate limitations from experiment data
limitation_data = {
    "test_set_size": 274,
    "trained_variants": ["v0", "v1", "v2"],
    "configured_variants": ["v0", "v1", "v2", "v3", "v4"],
    "dataset_source": "Roboflow Universe person-faces v5",
    "benchmark_environment": "WSL2"
}

limitations = suggest_limitations_from_experiment(limitation_data)
print("Identified limitations:")
for lim in limitations:
    print(f"- {lim['statement']}")
    print(f"  Impact: {lim['impact']}\n")
```

## Honest Reporting Examples

From our YOLOv11 face detection project:

| Statement | Type |
|---|---|
| "Target missed by 0.006 mAP points" | **Result**, not limitation — report it plainly |
| "Test set is small (274 images)" | **Limitation** — differences <1 pt are statistically indistinguishable |
| "V3 and V4 were not trained due to time" | **Incomplete search** — clearly labeled as such |

**Key principle:** Readers need to know what inferences they can/can't safely draw.
