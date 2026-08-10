# Data Profiling Skills

Profile data properties before making any architectural decisions — let data drive design, not dogma.

## Skills in this Category

### analysis
Analyze dataset distributions (e.g., bounding box scales in object detection) to determine which model components are actually needed.

**When to use:** Before removing heads, changing stride, modifying preprocessing  
**Key question answered:** "Is dropping P5 head justified by data?"  
**Output:** Distribution plots, quantitative tables of component necessity

## Example Usage

```python
from Skills.data_profiling.analysis import profile_box_sizes, recommend_head_removal

# Profile box sizes to decide which detection heads are needed
box_profile = profile_box_sizes(dataset_path="data/train/")

# Get recommendation
recommendation = recommend_head_removal(box_profile)
print(f"Can remove heads: {recommendation['_overall']['can_remove_any_head']}")
print(f"Evidence: {recommendation['_overall']['evidence']}")
```

## Philosophy

Measure before you modify. Never drop components based on heuristics without measuring the actual data distribution first.
