# Conversion Analysis Skills

Separate accuracy losses by export stage — often conversion itself costs more than quantisation.

## Skills in this Category

### boundary
Measure accuracy loss at each stage: PyTorch → ONNX/TFLite → INT8.

**When to use:** Any model export workflow  
**Sequence measured:**
1. FP32 native (baseline)
2. Exported FP32/FP16 (conversion loss)
3. Quantised INT8 (quantisation impact)

**Output:** Accuracy delta table, recommendation on whether quantisation is worth it

## Example Usage

```python
from Skills.conversion_analysis.boundary import analyze_conversion_pipeline

# Analyze conversion pipeline
results = analyze_conversion_pipeline(
    model_path="models/yolo11n-face-v2",
    test_split_path="data/test/"
)

# See where accuracy actually goes
print(f"Conversion loss (PyTorch→OpenVINO): {results['_deltas']['pytorch_to_conversion_loss']}")
print(f"Quantisation loss (FP32→INT8): {results['_deltas']['fp16_to_int8_quantisation_impact']}")

# Get recommendations
print(f"Recommendation: {results['_analysis']['recommendation']}")
```

## Research Insight

In our YOLOv11 face detection project:
- **Conversion loss:** -0.55 pt mAP (preprocessing differences)
- **Quantisation loss:** +0.09 pt (essentially free!)

**Conclusion:** Focus on fixing conversion boundary, not quantisation.
