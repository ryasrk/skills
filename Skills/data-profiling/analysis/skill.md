# Skill: Data Profiling - Distribution Analysis

**Profile data properties BEFORE making architectural decisions.** Don't drop heads/components based on dogma — measure actual data distributions first.

---

## 🎯 When to Use This Skill

Apply this skill **BEFORE ANY ARCHITECTURE MODIFICATION**:

- Considering dropping detection heads in YOLO/detection models
- Removing attention blocks from transformers  
- Changing stride patterns or receptive fields
- Modifying input preprocessing pipeline
- Reducing model width/channels

**Key insight:** Never delete components without measuring what fraction of your data actually uses them.

---

## 📋 Core Principles

### 1. Measure Before You Modify
❌ **Wrong:** Following standard practice "drop P5 for small objects"  
✅ **Correct:** Profile box scales → see if < 64px objects exist

### 2. Quantify Component Necessity
❌ **Wrong:** "I think we can remove this block"  
✅ **Correct:** "Only 3.2% of boxes fall outside this component's range"

### 3. Document Rationale
❌ **Wrong:** No explanation for changes  
✅ **Correct:** Every architecture change linked to measured data property

---

## 🔧 Step-by-Step Instructions

### Step 1: Profile Box Scale Distribution (Detection Models)

```python
def profile_box_sizes(dataset_path: str, output_dir: str = "docs") -> dict:
    """
    Measure bounding box sizes across all images.
    
    Determines which detection heads are actually needed by measuring
    what fraction of ground-truth boxes fall into each head's receptive band.
    """
    # Load dataset annotations (YOLO format: class x_center y_center w h)
    boxes_per_image = []
    
    for anno_file in Path(dataset_path).glob("*.txt"):
        with open(anno_file) as f:
            boxes = []
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    _, x_center, y_center, width, height = map(float, parts[:5])
                    boxes.append((width, height))  # normalized coordinates
            if boxes:
                avg_size = np.mean([max(w, h) for w, h in boxes])
                boxes_per_image.append(avg_size)
    
    if not boxes_per_image:
        raise ValueError("No boxes found in dataset")
    
    boxes_per_image = np.array(boxes_per_image)
    
    # Define head receptive bands at 640px input
    stride_8_band = (0, 64)      # P3: objects < 64px
    stride_16_band = (64, 128)   # P4: objects 64-128px  
    stride_32_band = (128, None) # P5: objects > 128px
    
    def distribution_stats(band_min, band_max):
        if band_max is None:
            filtered = boxes_per_image >= band_min
        else:
            filtered = (boxes_per_image >= band_min) & (boxes_per_image < band_max)
        
        count = int(np.sum(filtered))
        total = len(boxes_per_image)
        share = count / total * 100 if total > 0 else 0
        
        return {
            "count": count,
            "share_percent": round(share, 1),
            "mean_px": float(np.mean(boxes_per_image[filtered])) if np.any(filtered) else 0,
            "median_px": float(np.median(boxes_per_image[filtered])) if np.any(filtered) else 0,
            "p95_px": float(np.percentile(boxes_per_image[filtered], 95)) if np.any(filtered) else 0
        }
    
    return {
        "P3_stride_8": distribution_stats(*stride_8_band),
        "P4_stride_16": distribution_stats(*stride_16_band),
        "P5_stride_32": distribution_stats(*stride_32_band),
        "total_images": len(boxes_per_image),
        "total_boxes": len(boxes_per_image),
        "overall_mean_px": float(np.mean(boxes_per_image)),
        "global_min_px": float(np.min(boxes_per_image)),
        "global_max_px": float(np.max(boxes_per_image))
    }
```

### Step 2: Generate Visualizations

```python
def generate_distribution_plot(results: dict, output_path: str = "docs/box_scale_distribution.png"):
    """Create bar chart showing share of boxes per head."""
    import matplotlib.pyplot as plt
    
    labels = ["P3 (<64px)", "P4 (64-128px)", "P5 (>128px)"]
    shares = [results[f"P{i}_stride_{8*i}"]["share_percent"] for i in range(1, 4)]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, shares, color=['#3b82f6', '#8b5cf6', '#ec4899'])
    plt.ylabel("Share of Boxes (%)")
    plt.title("Box Scale Distribution by Detection Head")
    plt.ylim(0, max(shares) * 1.2)
    
    # Add percentage labels
    for bar, share in zip(bars, shares):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{share:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plot to: {output_path}")
```

### Step 3: Get Architecture Recommendations

```python
def recommend_head_removal(box_profile: dict) -> dict:
    """
    Analyze whether any detection head can be safely removed.
    """
    recommendations = {}
    
    for head_key in ["P3_stride_8", "P4_stride_16", "P5_stride_32"]:
        share = box_profile[head_key]["share_percent"]
        
        if share < 5:
            recommendations[head_key] = {
                "action": "consider_removing",
                "confidence": "high",
                "rationale": f"Only {share:.1f}% of boxes fall in this head's range"
            }
        elif share < 15:
            recommendations[head_key] = {
                "action": "monitor",
                "confidence": "medium", 
                "rationale": f"Low but non-negligible coverage ({share:.1f}%); test removal experimentally"
            }
        else:
            recommendations[head_key] = {
                "action": "retain",
                "confidence": "high",
                "rationale": f"Mandatory: {share:.1f}% of targets depend on this head"
            }
    
    # Overall recommendation
    all_mandatory = all(r["action"] == "retain" for r in recommendations.values())
    
    if all_mandatory:
        overall = {
            "can_remove_any_head": False,
            "recommendation": "DO NOT remove any detection head",
            "evidence": "Data shows load spread across all heads; dropping any would forfeit significant target coverage"
        }
    else:
        removable = [k for k, v in recommendations.items() if v["action"] == "consider_removing"]
        overall = {
            "can_remove_any_head": True,
            "candidate_heads": removable,
            "risk": f"Removing these heads forfeits ~{sum(box_profile[h]['share_percent'] for h in removable):.1f}% of boxes"
        }
    
    recommendations["_overall"] = overall
    return recommendations
```

### Step 4: Check Augmentation Already Applied

```python
def check_augmentation_already_applied(dataset_config: dict) -> dict:
    """
    Detect if dataset already has augmentation baked in (e.g., Roboflow processed).
    
    Warning signs:
    - Very large train split vs small unique image count
    - Dataset provider notes baked-in transforms
    """
    warnings = []
    
    splits = dataset_config.get("splits", {})
    unique_images = dataset_config.get("unique_source_images", 0)
    
    if splits.get("train_count", 0) > 2000 and unique_images > 0:
        ratio = splits["train_count"] / unique_images
        if ratio > 2:
            warnings.append({
                "type": "possible_baked_augmentation",
                "severity": "high",
                "message": f"Train split is {ratio:.1f}x larger than unique images; may have baked-in augmentation",
                "action": "Reduce custom augmentation aggression or disable default transforms"
            })
    
    return {
        "warnings": warnings,
        "training_recommendation": "Use light augmentation profile" if warnings else "Standard augmentation acceptable"
    }
```

---

## 💻 Example Command Sequences

### Running Box Scale Analysis

```bash
# 1. Run profiling script
cd src && python analyze_data.py

# Output includes:
# - Box scale distribution stats
# - docs/box_scale_distribution.png visualization
# - Recommendation table
```

**Sample output:**

```
Box Scale Distribution:
| Detection Head | Receptive Band | Share of Boxes |
|---|---|---|
| P3_stride_8 | < 64 px | 28.4% |
| P4_stride_16 | 64–128 px | 20.1% |
| P5_stride_32 | > 128 px | 51.6% |

Recommendation: DO NOT remove any detection head
Evidence: Dropping P5 forfeits 51.6% of targets; dropping P3 forfeits 28.4%
```

### For Time Series Data

```python
def profile_time_series_features(data_path: str) -> dict:
    """
    Profile time series for seasonal patterns, frequency content, missingness.
    """
    import pandas as pd
    from scipy import signal
    
    df = pd.read_csv(data_path)
    values = df['value'].values
    
    features = {
        "missing_percentage": float(np.isnan(values).sum() / len(values) * 100),
        "autocorrelation_lag_1": float(pd.Series(values).autocorr(lag=1)),
        "dominant_frequency_hz": estimate_dominant_freq(values),
        "trend_coefficient": fit_linear_trend(values)[0],
        "seasonality_amplitude": measure_seasonal_amplitude(values)
    }
    
    return features
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Dropping Heads Based on Dogma

**WRONG:** "Face detection means small objects → drop P5"  
**Following heuristics without measuring**

**CORRECT:** "Measure box scales first → P5 carries 51.6% of boxes → cannot drop"

From our research: Dropping P5 would forfeit **52%** of all targets — completely wrong intuition!

### ❌ Mistake 2: Ignoring Data Augmentation Effects

**WRONG:** Apply Ultralytics default augmentation to Roboflow dataset  
**Dataset already has baked-in augmentation**

**CORRECT:** Check `check_augmentation_already_applied()` → use light augmentation profile

### ❌ Mistake 3: Not Visualizing Distributions

**WRONG:** Just report numbers  
**Hard to communicate to team**

**CORRECT:** Generate plots like `box_scale_distribution.png` with percentage labels

### ❌ Mistake 4: Making Changes Without Data Proof

**WRONG:** "Let me just try removing C2PSA attention block"  
**Post-hoc rationalization after seeing results**

**CORRECT:** "C2PSA processes deepest stage; measure its FLOP share first (4%), then decide"

Note: In our case, removing C2PSA was fine BUT for different reason than initially thought — data showed it's not critical for single-class task with low faces/image average.

---

## ✅ Success Indicators

You've done proper data profiling when:

1. ✅ Generated distribution plots saved to `docs/`
2. ✅ Quantified share of data using each component
3. ✅ Have clear rationale for every architecture decision
4. ✅ Checked for baked-in augmentation effects
5. ✅ Can explain to stakeholders why certain heads/components must stay
6. ✅ All measurements reproducible from raw data

**Test yourself:** Could a reviewer challenge your architecture choices? If they need more data evidence, you haven't profiled enough.

---

## 🔗 Related Skills

| Skill | Purpose |
|---|---|
| `weight_transfer/accounting` | Track how much pretrained knowledge lost if changing widths |
| `limitations/documentation` | Document incomplete variant search due to data constraints |
| `experimental_design/constraints` | Frame profiling within deployment budget |

---

## 📚 Reference Examples

### From Face Detection Project

**Before profiling:** Thought we should drop P5 head for face detection  
**After profiling:** P5 carries 51.6% of boxes → MUST RETAIN ALL HEADS

**Key findings:**
```json
{
    "P3_share": 28.4,
    "P4_share": 20.1,
    "P5_share": 51.6,
    "min_box_px": 10,
    "max_box_px": 481,
    "recommendation": "No head removable - data spans all scales"
}
```

This single measurement redirected entire slimming effort away from head count onto width/channel modifications where savings were actually possible.

### What Changed After Learning This

1. Started profiling box scales before ALL future detection projects
2. Never follow "standard practices" without measuring own data first
3. Always visualize distributions — hard to argue with charts
4. Check for baked-in augmentation before stacking transforms

---

## 🎯 Quick Checklist

Before modifying ANY architecture component:

- [ ] Profiled relevant data distribution
- [ ] Generated quantitative table of component usage
- [ ] Created visualization for team communication
- [ ] Got explicit recommendation (retain/monitor/remove)
- [ ] Checked for baked-in data effects
- [ ] Documented rationale linking data → decision
- [ ] Measured alternative metrics (FLOPs, transfer rates) if applicable

If any checkbox is unchecked, don't modify that component yet.
