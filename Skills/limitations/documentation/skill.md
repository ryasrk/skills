# Skill: Limitations Documentation - Honest Constraint Reporting

**Explicitly document structural constraints that bound how far results generalise — distinguish limitations from incomplete searches, not just "things we didn't try."**

---

## 🎯 When to Use This Skill

Apply this skill **BEFORE PRESENTING OR PUBLISHING ANY RESEARCH**:

- Academic paper submission
- Technical report publication  
- Product demo with claims
- Conference presentation slides
- GitHub README with benchmark results
- ANY public documentation of experimental findings

**Key insight:** Readers need to know what inferences they can/can't safely draw from your work.

---

## 📋 Core Principles

### 1. Distinguish Three Types of Statements

**Limitation:** Cannot fix even with infinite compute/data → Bounds generalisability
**Incomplete search:** Could fix but didn't due to practical constraints → Unexplored space
**Result:** What you found, good or bad → Report it plainly

### 2. Every Limitation Must Answer "So What?"

❌ **Wrong:** "Test set is small"  
✅ **Correct:** "Test set is small (274 images) → differences <1 pt mAP are statistically indistinguishable"

### 3. Negative Results Are Findings Too

❌ **Wrong:** Hide target miss entirely  
✅ **Correct:** "Best model achieved 0.9494 mAP@0.5 vs 0.95 target — missed by 0.006 pts (~1/3 of one box)"

---

## 🔧 Step-by-Step Instructions

### Step 1: Categorize Each Statement Correctly

```python
def categorize_statement(statement_type: str, statement_text: str) -> dict:
    """
    Classify research statements into proper categories.
    
    Returns structured format for consistent reporting.
    """
    categories = {
        "limitation": {
            "definition": "Structural constraint that cannot be fixed",
            "examples": [
                ("statistical", "Small test set creates noise floor"),
                ("domain_transfer", "Single-source dataset limits generalisation"),
                ("environmental", "Benchmark environment differs from deployment")
            ]
        },
        "incomplete_search": {
            "definition": "Could have explored more but didn't",
            "examples": [
                ("untrained_variants", "V3 and V4 configured but not trained"),
                ("hyperparameters", "Did not sweep learning rate range fully")
            ]
        },
        "result": {
            "definition": "What was actually measured/found",
            "examples": [
                ("target_missed", "Achieved 0.9494 vs 0.95 target"),
                ("negative_control_passed", "8-thread control showed NO speedup")
            ]
        }
    }
    
    return {
        "type": category,
        "statement": statement_text,
        "impact_explanation": None,  # Will fill in below
        "evidence_source": None      # Point to specific data/files
    }
```

### Step 2: Generate Automatic Limitations Suggestions

```python
def suggest_limitations_from_experiment(experiment_metadata: dict) -> list:
    """
    Auto-suggest likely limitations based on experiment metadata.
    
    Checks common patterns that indicate structural constraints.
    """
    suggestions = []
    
    # Small test set → statistical uncertainty
    test_set_size = experiment_metadata.get("test_set_size", 0)
    if test_set_size < 500:
        suggestions.append({
            "category": "statistical",
            "statement": f"The test set is small ({test_set_size} images). Differences below ~{calculate_noise_floor(test_set_size):.2f} points of mAP@0.5 are statistically indistinguishable.",
            "impact": "Cannot confidently claim variants differ; sub-{calculate_noise_floor(test_set_size):.2f}-pt improvements may be noise",
            "evidence": f"{test_set_size} test images / {experiment_metadata['test_boxes']} boxes"
        })
    
    # Incomplete training run → search space gap
    trained_variants = experiment_metadata.get("trained_variants", [])
    configured_variants = experiment_metadata.get("configured_variants", [])
    untrained = set(configured_variants) - set(trained_variants)
    
    if untrained:
        suggestions.append({
            "category": "incomplete_search",
            "statement": f"Variant search incomplete — {' and '.join(untrained)} were configured but not trained due to time/compute constraints.",
            "impact": "Cannot claim best-found model is globally optimal; promising configurations remain untested",
            "evidence": f"Configured: {configured_variants}; Trained: {trained_variants}"
        })
    
    # Single dataset source → domain transfer limitation
    dataset_source = experiment_metadata.get("dataset_source", "")
    unique_sources = len(dataset_source.split(",")) if dataset_source else 0
    
    if unique_sources <= 1:
        suggestions.append({
            "category": "domain_transfer",
            "statement": f"Single-source dataset ({dataset_source}). No cross-domain validation against other benchmarks like WIDER FACE or different provider datasets.",
            "impact": "Generalisation to crowds, extreme poses, low light unmeasured; results may not translate",
            "evidence": f"All images from one provider with baked-in augmentation"
        })
    
    # Virtualized environment → environmental concern
    env = experiment_metadata.get("benchmark_environment", "").lower()
    if "wsl" in env or "virtual" in env or "docker" in env:
        suggestions.append({
            "category": "environmental",
            "statement": f"Benchmarks ran under virtualised environment ({experiment_metadata.get('benchmark_environment')}, which virtualises CPU topology.",
            "impact": "Absolute numbers may differ from real deployment hardware; hybrid CPUs (P-core/E-core) indistinguishable",
            "evidence": f"WSL2 guest sees uniform CPUs, no frequency or core-type data available"
        })
    
    # Hardware feature dependency
    has_feature = experiment_metadata.get("cpu_has_vnni", False)
    feature_name = experiment_metadata.get("feature_name", "AVX-VNNI")
    
    if has_feature:
        suggestions.append({
            "category": "hardware_dependency",
            "statement": f"{feature_name} speedup depends on {feature_name.lower()} support present on test host but absent on some edge CPUs.",
            "impact": f"Edge systems without {feature_name.lower()} will see smaller gains than measured",
            "evidence": f"Host has AVX2 + {feature_name}; baseline performance tied to these instructions"
        })
    
    # Target misses documented as result, not limitation
    actual_result = experiment_metadata.get("best_actual_result")
    target = experiment_metadata.get("accuracy_target")
    
    if actual_result and target and actual_result < target:
        diff = target - actual_result
        suggestions.append({
            "category": "result",
            "statement": f"Accuracy target ({target:.4f}) was missed. Best deployable model achieved {actual_result:.4f} — short by {diff:.4f} points.",
            "impact": f"Model does NOT meet stated requirement; use only if tolerance allows {diff*100:.1f}% margin",
            "evidence": f"Test split measurements: {actual_result:.4f} vs target {target:.4f}"
        })
    
    return suggestions
```

### Step 3: Generate Well-Structured Markdown

```python
def generate_limitations_section(limitations_data: list, output_format: str = "markdown") -> str:
    """
    Format limitations section properly for documentation.
    
    Organizes by category with clear labels and impact statements.
    """
    
    # Group by category
    by_category = {}
    for lim in limitations_data:
        cat = lim["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(lim)
    
    category_titles = {
        "statistical": "Statistical Uncertainty",
        "incomplete_search": "Incomplete Variant Search",
        "domain_transfer": "Domain Transfer Limits",
        "environmental": "Environmental Validity Concerns",
        "hardware_dependency": "Hardware Capability Dependencies"
    }
    
    lines = [
        "## Limitations\n",
        "Stated plainly because they bound how far these results generalise.\n",
        ""
    ]
    
    for cat, items in sorted(by_category.items()):
        title = category_titles.get(cat, cat.title().replace("_", " "))
        
        lines.append(f"### {title}\n")
        
        for i, lim in enumerate(items):
            item_num = i + 1
            
            # Quote their specific statement
            lines.append(f"{item_num}. **{lim['statement']}**\n")
            
            # Explain impact on inference
            if "impact" in lim and lim["impact"]:
                lines.append(f"   *Impact*: {lim['impact']}\n")
            
            # Show evidence
            if "evidence" in lim and lim["evidence"]:
                lines.append(f"   *Evidence*: {lim['evidence']}\n")
            
            lines.append("\n")
    
    return "\n".join(lines)
```

### Step 4: Review for Common Pitfalls

```python
def review_limitations_for_pitfalls(limitations: list) -> dict:
    """
    Check limitations section for common documentation mistakes.
    
    Returns warnings about problematic patterns.
    """
    warnings = []
    
    # Check for "we didn't try X" phrasing masquerading as limitations
    for lim in limitations:
        text = lim.get("statement", "").lower()
        
        if "didn't try" in text or "not tested" in text or "never attempted" in text:
            warnings.append({
                "severity": "medium",
                "message": f"Phrase '{text[:30]}...' sounds like incomplete search, not structural limitation",
                "suggestion": "Rephrase as 'Incomplete search' category instead"
            })
        
        if "small" in text and "dataset" in text:
            # Check if noise floor is mentioned
            impact = lim.get("impact", "").lower()
            if "noise" not in impact and "indistinguishable" not in impact:
                warnings.append({
                    "severity": "low", 
                    "message": "Small dataset limitation should explain statistical implications",
                    "suggestion": "Add: 'Differences below X points are statistically indistinguishable'"
                })
    
    # Check for overly vague claims
    vague_phrases = ["may", "might", "possibly", "seems"]
    for lim in limitations:
        text = lim.get("statement", "").lower()
        matches = [p for p in vague_phrases if p in text]
        if matches:
            warnings.append({
                "severity": "low",
                "message": f"Statement uses vague language: {', '.join(matches)}",
                "suggestion": "Be more precise with exact numbers and boundaries"
            })
    
    return {
        "warnings": warnings,
        "total_issues": len(warnings),
        "can_publish": len([w for w in warnings if w["severity"] == "high"]) == 0
    }
```

---

## 💻 Example Command Sequences

### Running Limitation Generator

```bash
# Collect experiment metadata first
python scripts/collect_experiment_metadata.py \
    --results-dir results/ \
    --output experiment_meta.json

# Generate limitations automatically
python scripts/generate_limitations.py \
    --metadata experiment_meta.json \
    --output docs/limitations.md

# Review for issues
python scripts/review_limitations.py \
    --input docs/limitations.md \
    --show-warnings
```

**Sample command flow:**
```bash
$ python scripts/generate_limitations.py --metadata experiment_meta.json

Generated limitations:

### Statistical Uncertainty

1. **The test set is small (274 images). Differences below ~0.01 points of mAP@0.5 are statistically indistinguishable.**
   *Impact*: Cannot confidently claim variants differ; sub-0.01-pt improvements may be noise
   *Evidence*: 274 test images / 463 boxes

### Incomplete Variant Search

2. **Variant search incomplete — yolo11n-face-v3 and yolo11n-face-v4 were configured but not trained due to time/compute constraints.**
   *Impact*: Cannot claim best-found model is globally optimal; promising configurations remain untested
   *Evidence*: Configured: v0,v1,v2,v3,v4; Trained: v0,v1,v2

### Domain Transfer Limits

3. **Single-source dataset (Roboflow Universe person-faces v5). No cross-domain validation against other benchmarks like WIDER FACE.**
   *Impact*: Generalisation to crowded scenes, extreme poses, low light unmeasured; results may not translate
   *Evidence*: All images from one provider with baked-in augmentation
```

### Manual Addition for Specific Issues

```python
# Add custom limitations not caught by auto-generator
custom_limitations = [
    {
        "category": "hardware_dependency",
        "statement": "INT8 speedup depends on AVX-VNNI support present on test host but absent on some edge CPUs.",
        "impact": "Edge systems without AVX-VNNI will see smaller INT8 gains than measured",
        "evidence": "Host has AVX2 + AVX-VNNI; baseline performance tied to these instructions"
    },
    {
        "category": "result", 
        "statement": "Accuracy target (mAP@0.5 ≥ 0.95) was missed. Best deployable model achieved 0.9494 — short by 0.0006 points.",
        "impact": "Model does NOT meet stated requirement; use only if tolerance allows 0.06% margin",
        "evidence": "Test split measurements: 0.9494 vs target 0.9500"
    }
]

with open("docs/limitations.md", "a") as f:
    f.write(generate_limitations_section(custom_limitations))
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Mistake 1: Calling Results Limitations

**WRONG:** "Target missed by 0.006 pts" labeled as limitation  
**This is a RESULT, not a limitation bounding generalisability**

**CORRECT:** Separate honestly:
```
Results Section:
  ✓ Target met in PyTorch: 0.9517
  ✓ Target met on val split: 0.9511
  ✗ Target missed on test split (INT8): 0.9494
  
Limitations Section:
  • Test set is small → statistical uncertainty
  • Variant search incomplete → could do better
```

### ❌ Mistake 2: Hiding Disappointing Results

**WRONG:** Only highlight where target WAS met (val split)  
**Hide failure on held-out test**

**CORRECT:** State plainly in both places:
- Results: "missed by 0.0006"
- Limitations: explains why this matters given test size

### ❌ Mistake 3: Not Explaining Impact

**WRONG:** "Test set is small"  
**Reader doesn't know what to conclude**

**CORRECT:** "Test set is small (274 images) → differences <1 pt mAP are statistically indistinguishable"

### ❌ Mistake 4: Blaming Environment Too Vaguely

**WRONG:** "Benchmarks ran on non-production hardware"  
**Too vague to be useful**

**CORRECT:** "WSL2 virtualises CPU topology; host is hybrid Intel Core Ultra 7 255HX but guest sees 12 uniform CPUs with no frequency/core-type data"

---

## ✅ Success Indicators

Your limitations section is production-ready when:

1. ✅ Clearly distinguishes limitations vs incomplete searches vs results
2. ✅ Every limitation has explicit impact statement ("so what?")
3. ✅ Cites specific evidence (numbers, files, conditions)
4. ✅ Negative results presented prominently (not hidden)
5. ✅ Can answer reviewer questions about each limitation immediately
6. ✅ No vague phrases like "may", "might", "possibly"
7. ✅ Structural constraints clearly identified as irreversible

**Test yourself:** If a skeptic read ONLY your limitations section, would they understand exactly what they CAN'T trust from your paper? If yes, you've got it right.

---

## 🔗 Related Skills

| Skill | Purpose |
|---|---|
| `metrics_pipeline/generation` | Store limiting experiment metadata alongside metrics |
| `environment_testing/validation` | Document environmental validity concerns |
| `data_profiling/analysis` | Include dataset scale limitations |

---

## 📚 Reference Examples

### From Face Detection Project

Our limitations section included:

**Statistical Limitations:**
```
"The gate was missed. 0.9494 vs 0.95 on test for the deployed INT8 model."
"The test set is small: 274 images / 463 boxes. Differences below ~1 point of mAP@0.5 are not statistically meaningful."
```

**Incomplete Search:**
```
"The variant search is incomplete. V3 (width + attention removal combined) and V4 (512 px) are configured in configs/models/ but were not trained; the run was stopped early for time."
```

**Domain Limits:**
```
"Single-source dataset. All images come from one Roboflow dataset with baked-in augmentation. No cross-domain validation against WIDER FACE, so generalisation to crowded scenes, extreme poses or low light is unmeasured."
```

**Environment Validity:**
```
"Benchmarks ran under WSL2, which virtualises CPU topology. The host is a hybrid Intel Core Ultra 7 255HX with P-cores and E-cores, but the guest sees 12 uniform CPUs with no core-type or frequency data."
```

**Hardware Dependencies:**
```
"INT8 speedup depends on VNNI. This host has AVX2 + AVX-VNNI. Edge CPUs without VNNI will see a materially smaller INT8 gain than the 2.3× measured here."
```

### What We Learned

1. Honesty builds credibility — readers respect transparent limitations
2. Distinguishing results from limitations prevents confusion
3. Specific evidence (274 images, 463 boxes) makes limitations credible
4. Not hiding misses actually strengthens overall narrative

---

## 🎯 Quick Checklist

Before publishing ANY experimental results:

- [ ] Listed ALL structural constraints affecting conclusions
- [ ] Separated limitations (immutable) from incomplete searches (practical)
- [ ] Explicitly stated impacts of each limitation
- [ ] Provided evidence citations for all claims
- [ ] Presented negative results/prominently (not hidden)
- [ ] Removed all vague language ("may", "might")
- [ ] Would satisfy a skeptical reviewer asking "what can I NOT trust?"
- [ ] Committed limitations text alongside metrics in repository

If any checkbox is unchecked, you're not ready to publish.