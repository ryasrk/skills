# Skill: Weight Transfer - Parameter Accounting

**Track how many pretrained parameters actually transfer when modifying architectures.** Parameter counts lie; weight inheritance reveals hidden costs of architecture changes.

---

## 🎯 When to Use This Skill

Apply this skill whenever **modifying model architecture**:

- Scaling width multipliers (0.25 → 0.1875)
- Removing blocks/heads from networks  
- Adding custom layers on top of backbones
- Changing stride patterns or channel dimensions
- Any modification to pretrained models on < 10k samples

**Key insight:** A 40% parameter reduction can still hurt accuracy if you're learning most parameters from scratch on small datasets.

---

## 📋 Core Principles

### 1. Measure Transfer Rate, Not Just Param Count
❌ **Wrong:** "V1 has 60% fewer params than V0"  
✅ **Correct:** "V1 only inherits 24% of weights — must learn 76% from scratch"

### 2. Small Datasets Depend on Pretrained Knowledge
❌ **Wrong:** Training on 800 images with randomized initialization  
✅ **Correct:** Keep architecture close to stock for high transfer rates

### 3. Channel Changes Break Transfer More Than Head Removals
❌ **Wrong:** Modifying width multipliers thinking it's safe  
✅ **Correct:** Removing heads with `nn.Identity()` preserves alignment better

---

## 🔧 Step-by-Step Instructions

### Step 1: Implement Weight Transfer Checker

```python
def check_weight_transfer(model_variant_path: str, pretrained_checkpoint: str) -> dict:
    """
    Measure % of model's parameters that transfer from pretrained checkpoint.
    
    This reveals the hidden cost of architecture modifications:
    - Reducing width changes channel counts → most tensors can't transfer
    - Keeping topology identical → most weights transfer
    - Simply removing blocks with Identity() preserves index alignment
    
    Args:
        model_variant_path: Path to variant YAML/config
        pretrained_checkpoint: Path to pretrained checkpoint (e.g., coco.pt)
    
    Returns:
        Transfer metrics including per-layer statistics and overall rate
    """
    # Load variant model architecture
    # For YOLOv11/Ultralytics:
    from ultralytics import YOLO
    model = YOLO(f"configs/models/{model_variant_path}.yaml")
    
    # Load pretrained checkpoint
    state_dict_pretrained = torch.load(pretrained_checkpoint, map_location='cpu')['state_dict']
    
    # Check overlap with PRETRAINED checkpoint (not self!)
    # Correct method: intersect modified model against pretrained checkpoint
    transferred_keys = []
    reinitialized_keys = []
    
    for key in model.state_dict().keys():
        if key in state_dict_pretrained:
            # Also check shape compatibility
            if model.state_dict()[key].shape == state_dict_pretrained[key].shape:
                transferred_keys.append(key)
            else:
                reinitialized_keys.append(f"{key} (shape mismatch)")
        else:
            reinitialized_keys.append(key)
    
    total_params = sum(p.numel() for p in model.parameters())
    transferred_params = sum(p.numel() for name, p in model.named_parameters() 
                            if name in transferred_keys)
    
    transfer_rate = transferred_params / total_params * 100 if total_params > 0 else 0
    
    return {
        "total_parameters": total_params,
        "transferred_parameters": transferred_params,
        "reinitialized_parameters": total_params - transferred_params,
        "transfer_rate_percent": round(transfer_rate, 1),
        "by_category": analyze_by_category(model.state_dict(), transferred_keys),
        "warning_threshold_exceeded": transfer_rate < 80,
        "recommendation": get_recommendation(transfer_rate),
        "example_transferred_keys": list(dict.fromkeys(transferred_keys))[:10],  # First 10 unique
        "example_reinitialized_keys": list(dict.fromkeys(reinitialized_keys))[:10]
    }
```

### Step 2: Analyze Transfer by Layer Category

```python
def analyze_by_category(model_state_dict, transferred_keys: set) -> dict:
    """Break down transfer by layer category."""
    categories = defaultdict(lambda: {"total": 0, "transferred": 0, "params_total": 0, "params_transferred": 0})
    
    for key, param in model_state_dict.items():
        if "conv" in key and "head" not in key:
            cat = "backbone_conv"
        elif "bn" in key or "norm" in key:
            cat = "batch_norm"
        elif ".m." in key or "attention" in key or "proj" in key:
            cat = "attention"
        elif "cv" in key and "detect" not in key and "head" not in key:
            cat = "neck_spp"
        elif "head" in key or "detect" in key or "cv3" in key:
            cat = "detection_heads"
        else:
            cat = "other"
        
        categories[cat]["total"] += 1
        categories[cat]["params_total"] += param.numel()
        
        if key in transferred_keys:
            categories[cat]["transferred"] += 1
            categories[cat]["params_transferred"] += param.numel()
    
    result = {}
    for cat, counts in categories.items():
        if counts["total"] > 0:
            rate = counts["params_transferred"] / counts["params_total"] * 100 if counts["params_total"] > 0 else 0
            result[cat] = {
                "total_layers": counts["total"],
                "transferred_layers": counts["transferred"],
                "rate_percent": round(rate, 1),
                "params_total_millions": round(counts["params_total"] / 1e6, 2),
                "params_transferred_millions": round(counts["params_transferred"] / 1e6, 2)
            }
    
    return result
```

### Step 3: Get Recommendations Based on Transfer Rate

```python
def get_recommendation(transfer_rate: float) -> str:
    """Generate recommendation based on transfer rate."""
    if transfer_rate >= 80:
        return f"✓ Good: {transfer_rate:.1f}% weights transfer; pretrained knowledge preserved"
    elif transfer_rate >= 50:
        return f"⚠ Warning: Only {transfer_rate:.1f}% weights transfer; consider keeping closer to stock"
    else:
        return f"✗ High risk: Only {transfer_rate:.1f}% weights transfer; architecture changes may hurt more than help on small datasets"


def print_transfer_table(variants_results: list) -> str:
    """Print formatted comparison table across variants."""
    lines = [
        "| Variant | Params | vs Stock | GFLOPs @640 | Pretrained tensors inherited |",
        "|---|---|---|---|---|"
    ]
    
    for v in variants_results:
        params = f"{v['params']:,}"
        vs_stock = f"{v.get('vs_stock_percent', 100)}%"
        flops = f"{v.get('gflops_640', 0):.3f}"
        inherited = f"{v.get('transferred_count', 0)}/{v.get('total_count', 0)} ({v['transfer_rate']}%)"
        
        line = f"| `{v['variant_name']}` | {params} | {vs_stock} | {flops} | {inherited} |"
        lines.append(line)
    
    return "\n".join(lines)
```

---

## 💻 Example Command Sequences

### Running Transfer Analysis

```bash
# 1. Run comparison script
cd src && python compare_architectures.py

# Output includes:
# - Transfer rate table
# - Per-category breakdown
# - Warnings if transfer rate too low
```

**Sample output:**

```
Architecture Comparison:
| Variant | Params | vs Stock | GFLOPs @640 | Pretrained tensors inherited |
|---|---|---|---|---|
| yolo11n-face-v0 | 2,590,035 | 100% | 6.500 | 448/499 (89.8%) |
| yolo11n-face-v1 | 1,564,599 | 60% | 4.241 | 120/499 (24.0%) ⚠️
| yolo11n-face-v2 | 2,340,307 | 90% | 6.236 | 406/457 (88.8%) |
| yolo11n-face-v3 | 1,423,383 | 55% | 4.080 | 113/457 (24.7%) ⚠️

Key insight:
- V0/V2 inherit ~89% of tensors from COCO
- V1/V3 inherit ~24% — width scaling changes almost every channel count
- This explains why V1 has 40% fewer params but 1.6 mAP points worse accuracy
```

### Export Results to JSON

```bash
# Save analysis results for documentation
python scripts/export_architecture_analysis.py \
    --variants v0 v1 v2 v3 \
    --output results/architecture.json
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Optimizing Parameters Without Checking Transfer

**WRONG:** Reduce width multiplier to save params → train on small dataset  
**Result:** Model learns random weights instead of leveraging pretrained knowledge

**CORRECT:** Check transfer rate first → keep width near stock for small datasets

From our research:
- V1 (width scaled): 1.6 mAP lower than V0 despite 40% fewer params
- Cause: Had to learn 76% of network from scratch on 821 unique images

### ❌ Mistake 2: Deleting Heads Instead of Using Identity

**WRONG:** Delete C2PSA block entirely → all subsequent indices shift
**Corrupts weight transfer mapping accidentally**

**CORRECT:** Replace with `nn.Identity` → preserves downstream index alignment

This ensured V2 maintained 88.8% transfer rate vs corruption a naive deletion would have introduced.

### ❌ Mistake 3: Comparing Self Against Self (Not Pretrained Checkpoint)

**WRONG:** Load model → modify → compare state_dict before vs after  
**Always shows 100% transfer because shapes never change**

**CORRECT:** Compare MODIFIED model against ORIGINAL pretrained checkpoint

We initially made this error — our corrected version reports the real 24% figure.

### ❌ Mistake 4: Ignoring Per-Category Differences

**WRONG:** Just report overall transfer rate  
**Missing which components don't transfer well**

**CORRECT:** Break down by backbone/head/neck → see where knowledge is lost

In YOLOv11 case: detection heads had 100% transfer even with minor topology changes, but conv layers had varying rates.

---

## ✅ Success Indicators

You've properly tracked weight transfer when:

1. ✅ Measured against pretrained checkpoint (not self-comparison)
2. ✅ Accounted for shape compatibility (not just key existence)
3. ✅ Broke down by layer category
4. ✅ Saved results to `results/architecture.json` for reproducibility
5. ✅ Can explain why certain variants perform differently
6. ✅ Have warnings for low transfer rates (< 80%)

**Test yourself:** Can you predict performance degradation purely from transfer rates? In our case: 24% transfer → ~1.6 mAP loss prediction matched actual results.

---

## 🔗 Related Skills

| Skill | Purpose |
|---|---|
| `data_profiling/analysis` | Profile data first before deciding modifications |
| `limitations/documentation` | Document incomplete searches due to transfer constraints |
| `experimental_design/constraints` | Balance transfer vs deployment budget |

---

## 📚 Reference Examples

### From Face Detection Project

**Architectural variants tested:**
- **V0 (stock):** 89.8% transfer → meets target mAP@0.5 = 0.9517
- **V1 (width scaled):** 24.0% transfer → misses target mAP@0.5 = 0.9362 (1.5 pts lower)
- **V2 (C2PSA removed via Identity):** 88.8% transfer → meets target mAP@0.5 = 0.9539 (slightly better!)
- **V3 (width + attention):** 24.7% transfer → misses target mAP@0.5 = 0.9343

**Key finding:** Width scaling destroys transfer ability on small datasets regardless of FLOP savings.

### What We Learned

1. Keep architecture close to stock when training on < 5k unique images
2. Remove components with `nn.Identity` not deletion to preserve alignment
3. Transfer rate predicts accuracy penalty much better than param count alone
4. Always measure transfer explicitly — don't assume "minor" changes are free

---

## 🎯 Quick Checklist

Before committing ANY architecture modification:

- [ ] Compared modified model against pretrained checkpoint (not self)
- [ ] Checked shape compatibility for each tensor
- [ ] Calculated overall transfer rate percentage
- [ ] Broke down transfer by layer category
- [ ] Warned if transfer < 80% for small datasets
- [ ] Saved results to JSON file
- [ ] Can explain expected accuracy impact from transfer rate
- [ ] Considered alternative with higher transfer if possible

If any checkbox is unchecked, reconsider the modification.