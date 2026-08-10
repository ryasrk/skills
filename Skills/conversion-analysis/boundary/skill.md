# Skill: Conversion Analysis - Export Boundary Tracing

**Separate accuracy/speed losses by export stage.** Often PyTorch→ONNX/INT8 conversion costs more than quantisation itself — know where to focus efforts.

---

## 🎯 When to Use This Skill

Apply this skill whenever **exporting models to deployment formats**:

- PyTorch → ONNX / OpenVINO / TensorRT
- TensorFlow → TFLite / TF.js  
- Any framework → intermediate format → final deployment target
- Considering whether INT8 quantisation is worth the pipeline complexity

**Key insight:** Don't assume quantisation kills accuracy — measure conversion loss FIRST.

---

## 📋 Core Principles

### 1. Measure Accuracy at Each Stage
❌ **Wrong:** Compare FP32 native vs INT8 only  
✅ **Correct:** Trace FP32 native → ONNX FP32 → INT8 separately

### 2. Conversion Loss ≠ Quantisation Loss
❌ **Wrong:** "Quantisation cost us mAP"  
✅ **Correct:** "Conversion preprocessing difference cost us -0.55 pt; INT8 was essentially free"

### 3. Speed Gains Should Be Measured Per Format
❌ **Wrong:** Just report FPS improvement  
✅ **Correct:** FP32 vs FP16 (often no difference on CPU) vs INT8 (large speedup)

---

## 🔧 Step-by-Step Instructions

### Step 1: Define Standard Measurement Sequence

```python
def analyze_conversion_pipeline(model_path: str, test_split_path: str) -> dict:
    """
    Measure accuracy loss at each export stage in sequence.
    
    Sequence:
    1. FP32 native PyTorch (baseline)
    2. Exported FP32/FP16 (conversion loss)
    3. Quantised INT8 (quantisation impact)
    
    Returns delta for each step, showing where accuracy actually goes.
    """
    results = {}
    
    # Step 1: Native evaluation
    print("Step 1: Evaluating native model...")
    metrics_native = evaluate_native_model(model_path, test_split_path)
    results["native_fp32"] = {
        "mAP@0.5": metrics_native["map50"],
        "mAP@0.5:0.95": metrics_native["map5095"],
        "precision": metrics_native["precision"],
        "recall": metrics_native["recall"],
        "fps": metrics_native["fps"]
    }
    
    # Step 2: Export to ONNX/TensorRT/OpenVINO
    print("Step 2: Exporting to OpenVINO FP32...")
    export_to_openvino(model_path, precision="fp32", output_path=model_path + "/openvino_fp32/")
    
    metrics_fp32_ov = evaluate_openvino_model(model_path + "/openvino_fp32/", test_split_path)
    results["openvino_fp32"] = {
        "mAP@0.5": metrics_fp32_ov["map50"],
        "mAP@0.5:0.95": metrics_fp32_ov["map5095"],
        "precision": metrics_fp32_ov["precision"],
        "recall": metrics_fp32_ov["recall"],
        "fps": metrics_fp32_ov["fps"]
    }
    
    # Step 3: Export FP16 (often same as FP32 on CPU)
    print("Exporting to OpenVINO FP16...")
    export_to_openvino(model_path, precision="fp16", output_path=model_path + "/openvino_fp16/")
    
    metrics_fp16_ov = evaluate_openvino_model(model_path + "/openvino_fp16/", test_split_path)
    results["openvino_fp16"] = {
        "mAP@0.5": metrics_fp16_ov["map50"],
        "mAP@0.5:0.95": metrics_fp16_ov["map5095"],
        "precision": metrics_fp16_ov["precision"],
        "recall": metrics_fp16_ov["recall"],
        "fps": metrics_fp16_ov["fps"]
    }
    
    # Step 4: Post-training quantisation to INT8
    print("Quantising to OpenVINO INT8...")
    export_to_openvino_quantized(model_path, calibration_data=test_split_path[:300], 
                                 output_path=model_path + "/openvino_int8/")
    
    metrics_int8 = evaluate_openvino_model(model_path + "/openvino_int8/", test_split_path)
    results["openvino_int8"] = {
        "mAP@0.5": metrics_int8["map50"],
        "mAP@0.5:0.95": metrics_int8["map5095"],
        "precision": metrics_int8["precision"],
        "recall": metrics_int8["recall"],
        "fps": metrics_int8["fps"]
    }
    
    # Calculate deltas
    results["_deltas"] = calculate_deltas(results)
    results["_analysis"] = interpret_deltas(results["_deltas"])
    
    return results
```

### Step 2: Calculate Accurate Deltas

```python
def calculate_deltas(metrics_by_stage: dict) -> dict:
    """Calculate accuracy deltas between stages."""
    p0 = metrics_by_stage["native_fp32"]["mAP@0.5"]
    p1 = metrics_by_stage["openvino_fp32"]["mAP@0.5"]
    p2 = metrics_by_stage["openvino_fp16"]["mAP@0.5"]
    p3 = metrics_by_stage["openvino_int8"]["mAP@0.5"]
    
    return {
        "pytorch_to_conversion_loss": round(p1 - p0, 4),  # Conversion boundary cost
        "fp32_to_fp16_impact": round(p2 - p1, 4),           # Precision change
        "fp16_to_int8_quantisation_impact": round(p3 - p2, 4),  # Quantisation cost
        "total_conversion_and_quantisation_loss": round(p3 - p0, 4)
    }


def calculate_speedup_factors(fps_by_stage: dict) -> dict:
    """Calculate speed improvements per stage."""
    fp32_fps = fps_by_stage.get("openvino_fp32", {}).get("fps", 1)
    fp16_fps = fps_by_stage.get("openvino_fp16", {}).get("fps", 1)
    int8_fps = fps_by_stage.get("openvino_int8", {}).get("fps", 1)
    
    return {
        "fp16_vs_fp32_factor": round(fp16_fps / fp32_fps, 2),
        "int8_vs_fp32_factor": round(int8_fps / fp32_fps, 2),
        "int8_vs_fp16_factor": round(int8_fps / fp16_fps, 2),
        "interpretation": get_speedup_interpretation(fp16_fps / fp32_fps, int8_fps / fp32_fps)
    }


def get_speedup_interpretation(fp16_vs_fp32, int8_vs_fp32) -> str:
    """Explain what speedups mean."""
    if int8_vs_fp32 >= 2.0:
        return f"INT8 provides {int8_vs_fp32:.1f}× speedup — major gain from quantisation"
    elif fp16_vs_fp32 > 1.05:
        return f"FP16 provides {fp16_vs_fp32:.2f}× speedup — some benefit but not dominant"
    else:
        return f"FP16 ≈ FP32 ({fp16_vs_fp32:.2f}×) — no benefit on CPU, use INT8 only for real gains"
```

### Step 3: Interpret Results

```python
def interpret_deltas(deltas: dict) -> dict:
    """Interpret what the deltas mean for deployment decisions."""
    analysis = {}
    
    conversion_loss = abs(deltas["pytorch_to_conversion_loss"])
    quant_loss = abs(deltas["fp16_to_int8_quantisation_impact"])
    
    if conversion_loss > 0.01:  # > 1 pt loss
        analysis["main_accuracy_concern"] = "conversion_boundary"
        analysis["recommendation"] = f"Focus efforts on fixing {conversion_loss:.3f} pt conversion loss, not quantisation"
        analysis["rationale"] = "PyTorch→ONNX preprocessing differences are bigger problem than INT8 numerics"
        
    elif quant_loss > 0.005:  # > 0.5 pt loss  
        analysis["main_accuracy_concern"] = "quantisation_sensitivity"
        analysis["recommendation"] = "Model sensitive to INT8; consider per-channel calibrations or fine-tuning after quant"
        analysis["rationale"] = "Quantisation is costing meaningful accuracy"
        
    else:
        analysis["main_accuracy_concern"] = "quantisation_safe"
        analysis["recommendation"] = "INT8 quantisation safe — deploy it"
        analysis["rationale"] = "Both conversion and quant loss < noise floor; INT8 delivers pure speed/size gains"
    
    analysis["speedup_factor"] = deltas.get("int8_vs_fp32_speedup", 0)
    analysis["file_size_reduction"] = estimate_file_size_reduction()
    
    return analysis
```

### Step 4: Generate Comparison Table

```python
def generate_comparison_table(results: dict) -> str:
    """Generate formatted accuracy comparison table for README."""
    lines = [
        "| Variant | Format | Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |",
        "|---|---|---|---|---|---|---|"
    ]
    
    format_mapping = {
        "native_fp32": "TORCH",
        "openvino_fp32": "FP32",
        "openvino_fp16": "FP16",
        "openvino_int8": "INT8"
    }
    
    for stage, format_name in format_mapping.items():
        data = results[stage]
        gate_mark = " ✅" if data["mAP@0.5"] >= 0.95 else " ❌"
        line = f"| `model` | {format_name} | test | {data['mAP@0.5']:.4f}{gate_mark} | {data['mAP@0.5:0.95']:.4f} | {data['precision']:.4f} | {data['recall']:.4f} |"
        lines.append(line)
    
    return "\n".join(lines)
```

---

## 💻 Example Command Sequences

### Running Full Conversion Analysis

```bash
# Execute conversion tracing script
python src/analyze_conversion.py \
    --model models/yolo11n-face-v2/train_runs/checkpoints/best.pt \
    --test-split data/test.json \
    --output results/conversion_analysis.json
```

**Sample output:**

```
=== CONVERSION ANALYSIS ===
Stage-by-stage accuracy:
├─ PyTorch FP32 baseline    : mAP@0.5 = 0.9539
├─ OpenVINO FP32            : mAP@0.5 = 0.9484   (-0.0055 conversion loss)
├─ OpenVINO FP16            : mAP@0.5 = 0.9485   (+0.0001 from FP32)
└─ OpenVINO INT8            : mAP@0.5 = 0.9494   (+0.0009 from FP16)

Deltas:
├─ pytorch_to_conversion_loss: -0.0055 (5.5 pt loss at export boundary)
├─ fp32_to_fp16_impact       : +0.0001 (essentially free)
└─ fp16_to_int8_quantisation_impact: +0.0009 (even slightly better!)

Recommendation: INT8 quantisation safe — deploy it for 2.3× speedup without accuracy penalty
Main concern: Conversion boundary (-0.55 pt) is where accuracy actually dies, not quantisation
```

### Export Model Pipeline

```bash
# Automated export sequence with benchmarking
./scripts/export_all_formats.sh \
    --model trained_model.pt \
    --formats fp32,fp16,int8 \
    --calibration-data data/calib_300_images.json
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Assuming Quantisation Kills Accuracy

**WRONG:** "Let's skip INT8 because quantisation usually hurts"  
**Assumption without measurement**

**CORRECT:** Run full pipeline → discover INT8 was +0.09 pt improvement!

From our research: INT8 didn't cost anything because the -0.55 pt loss happened at export time (preprocessing differences), not from quantisation.

### ❌ Mistake 2: Ignoring FP16 on CPU

**WRONG:** Report FP16 has "half size advantage"  
**FP16 often upconverted to FP32 by CPU plugins anyway**

**CORRECT:** Benchmark FP16 performance alongside FP32 → confirm no latency benefit

We found FP16 saved disk space but had identical latency to FP32 on CPU because OpenVINO upconverts at compile time.

### ❌ Mistake 3: Blaming Quantisation for All Losses

**WRONG:** "Our accuracy dropped 0.5 pt with quantisation"  
**Actually -0.55 pt at PyTorch→OpenVINO conversion boundary**

**CORRECT:** Trace losses precisely → focus fix efforts on correct bottleneck

The preprocessing differences between Ultralytics' letterbox and exported graph caused conversion loss, not numerics.

### ❌ Mistake 4: Not Using Adequate Calibration Data

**WRONG:** Use tiny calibration set → poor INT8 scale estimation  
**Can artificially inflate quantisation loss**

**CORRECT:** Use full training split or representative sample (we used 2,463 images, not 300 guidance)

---

## ✅ Success Indicators

You've properly traced conversion boundaries when:

1. ✅ Measured all four stages: native FP32 → exported FP32 → FP16 → INT8
2. ✅ Calculated precise deltas for each transition
3. ✅ Can explain why conversion costs accuracy (preprocessing mismatch, resize interpolation, etc.)
4. ✅ Verified INT8 is actually beneficial or harmful before deciding
5. ✅ Saved detailed JSON report for reproducibility
6. ✅ Have clear recommendation based on data, not assumptions

**Test yourself:** Could you tell exactly which export stage caused your biggest accuracy loss? If not, you haven't measured enough.

---

## 🔗 Related Skills

| Skill | Purpose |
|---|---|
| `metrics_pipeline/generation` | Store all stage results in JSON |
| `environment_testing/validation` | Benchmark each format under same constraints |
| `limitations/documentation` | Document if conversion loss can't be fixed |

---

## 📚 Reference Examples

### From Face Detection Project

**Measured accuracy trajectory:**
```
PyTorch FP32  : 0.9539 mAP@0.5 (baseline)
  ↓ -0.0055 pt (export conversion)
OpenVINO FP32 : 0.9484 mAP@0.5
  ↓ +0.0001 pt (precision change, negligible)
OpenVINO FP16 : 0.9485 mAP@0.5
  ↓ +0.0009 pt (quantisation, essentially free)
OpenVINO INT8 : 0.9494 mAP@0.5
```

**Speed improvements:**
```
FP32: 31.8 ms inference → 31.3 FPS end-to-end
FP16: 30.7 ms inference → 31.3 FPS end-to-end  (same as FP32!)
INT8: 13.2 ms inference → 68.3 FPS end-to-end  (2.3× faster)
```

**Conclusion:** Convertion loss (-0.55 pt) happens at PyTorch→OpenVINO boundary due to preprocessing differences, NOT from quantisation. INT8 is a pure win.

### What We Learned

1. Always measure conversion loss explicitly — don't blame quantisation
2. FP16 on CPU is only useful for file size, not performance
3. INT8 is typically free on modern hardware with VNNI support
4. Fix preprocessing mismatches if you need that lost accuracy back

---

## 🎯 Quick Checklist

Before deploying ANY exported model:

- [ ] Evaluated native FP32 baseline first
- [ ] Exported and evaluated FP32 version
- [ ] Exported and evaluated FP16 version  
- [ ] Exported and evaluated INT8 version
- [ ] Calculated accuracy deltas between all stages
- [ ] Calculated speed/fps deltas between all stages
- [ ] Identified main source of accuracy loss (conversion vs quantisation)
- [ ] Saved complete analysis to JSON file
- [ ] Made deployment decision based on measurements, not assumptions

If any checkbox is unchecked, you're flying blind.