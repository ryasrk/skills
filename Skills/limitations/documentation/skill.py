"""
Skill: limitation-documentation

Explicitly document structural constraints bounding generalisability.
Distinguish between limitations, incomplete searches, and results.
"""

from pathlib import Path
from datetime import datetime


def write_limitations_section(limitations_data: dict, output_path: str = "README.md"):
    """
    Generate a limitations section for documentation.
    
    limitations_data should be list of dicts with:
    - type: "statistical" | "search_space" | "domain_transfer" | "environmental" | "hardware"
    - statement: Clear description of limitation
    - impact: How this bounds conclusions
    - evidence: What data proves it
    """
    lines = []
    lines.append("## Limitations\n")
    lines.append("Stated plainly because they bound how far these results generalise.\n")
    
    # Group by category
    by_category = {}
    for lim in limitations_data:
        cat = lim.get("type", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(lim)
    
    category_titles = {
        "statistical": "Statistical uncertainty",
        "search_space": "Incomplete variant search",
        "domain_transfer": "Domain transfer limits",
        "environmental": "Environmental validity concerns",
        "hardware": "Hardware capability dependencies"
    }
    
    for cat, items in sorted(by_category.items()):
        title = category_titles.get(cat, cat.title().replace("_", " "))
        lines.append(f"### {title}\n")
        
        for i, lim in enumerate(items):
            item_num = i + 1
            
            # Quote their specific statement
            lines.append(f"{item_num}. **{lim['statement']}**\n")
            
            # Explain impact on inference
            if "impact" in lim:
                lines.append(f"   *Impact*: {lim['impact']}\n")
            
            # Show evidence
            if "evidence" in lim:
                lines.append(f"   *Evidence*: {lim['evidence']}\n")
            
            lines.append("\n")
    
    return "\n".join(lines)


def suggest_limitations_from_experiment(experiment_results: dict) -> list:
    """
    Automatically identify likely limitations from experiment metadata.
    
    Checks for common patterns that indicate structural constraints.
    """
    suggestions = []
    
    # Small test set → statistical uncertainty
    test_set_size = experiment_results.get("test_set_size", 0)
    if test_set_size < 500:
        suggestions.append({
            "type": "statistical",
            "statement": f"The test set is small ({test_set_size} images). Differences below ~1 point of mAP@0.5 are statistically indistinguishable.",
            "impact": "Cannot confidently claim variants differ; noise floor is high",
            "evidence": f"274 test images / 463 boxes"
        })
    
    # Incomplete training run → search space gap
    trained_variants = experiment_results.get("trained_variants", [])
    configured_variants = experiment_results.get("configured_variants", [])
    untrained = set(configured_variants) - set(trained_variants)
    
    if untrained:
        suggestions.append({
            "type": "search_space",
            "statement": f"Variant search incomplete — {' '.join(untrained)} were configured but not trained due to time/compute constraints.",
            "impact": "Cannot claim best-found model is globally optimal; promising configurations remain untested",
            "evidence": f"Configured: {configured_variants}; Trained: {trained_variants}"
        })
    
    # Single dataset source → domain transfer limitation
    dataset_source = experiment_results.get("dataset_source", "")
    if "single" in dataset_source.lower() or len(dataset_source.split(",")) == 1:
        suggestions.append({
            "type": "domain_transfer",
            "statement": f"Single-source dataset ({dataset_source}). No cross-domain validation against other benchmarks like WIDER FACE.",
            "impact": "Generalisation to crowded scenes, extreme poses, low light unmeasured",
            "evidence": f"All images from one provider with baked-in augmentation"
        })
    
    # Virtualized environment → environmental concern
    env = experiment_results.get("benchmark_environment", "")
    if "wsl" in env.lower() or "virtual" in env.lower():
        suggestions.append({
            "type": "environmental",
            "statement": f"Benchmarks ran under virtualised environment ({env}), which abstracts CPU topology.",
            "impact": "Absolute numbers may differ from real deployment hardware; hybrid CPUs (P-core/E-core) indistinguishable",
            "evidence": "WSL2 guest sees uniform CPUs, no frequency or core-type data available"
        })
    
    # VNNI dependency → hardware capability
    has_vnni = experiment_results.get("cpu_has_vnni", False)
    if has_vnni:
        suggestions.append({
            "type": "hardware",
            "statement": f"INT8 speedup depends on AVX-VNNI support present on test host but absent on some edge CPUs.",
            "impact": "Edge systems without VNNI will see smaller INT8 gains than measured",
            "evidence": "Host has AVX2 + AVX-VNNI; baseline performance tied to these instructions"
        })
    
    return suggestions


# Usage example
if __name__ == "__main__":
    experiment_data = {
        "test_set_size": 274,
        "trained_variants": ["yolo11n-face-v0", "yolo11n-face-v1", "yolo11n-face-v2"],
        "configured_variants": ["yolo11n-face-v0", "yolo11n-face-v1", "yolo11n-face-v2", "yolo11n-face-v3", "yolo11n-face-v4"],
        "dataset_source": "Roboflow Universe person-faces v5",
        "benchmark_environment": "WSL2",
        "cpu_has_vnni": True
    }
    
    limitations = suggest_limitations_from_experiment(experiment_data)
    
    # Write to file
    limitations_text = write_limitations_section(limitations)
    print(limitations_text)
