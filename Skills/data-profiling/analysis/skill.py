"""
Skill: data-profiling-before-modification

Profile data properties before making architectural decisions.
For object detection: measure box scale distributions per detection head's receptive band.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def profile_box_sizes(dataset_path: str, output_dir: str = "docs") -> dict:
    """
    Profile bounding box sizes in a YOLO/detection dataset.
    
    Determines which detection heads are actually needed by measuring
    what fraction of ground-truth boxes fall into each head's receptive band.
    """
    # Load dataset annotations (YOLO format or COCO)
    boxes_per_image = []
    
    for anno_file in Path(dataset_path).glob("*.txt"):  # or *.json for COCO
        with open(anno_file) as f:
            boxes = []
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    boxes.append((width, height))  # normalized coordinates
            if boxes:
                avg_size = np.mean([max(w, h) for w, h in boxes])
                boxes_per_image.append(avg_size)
    
    if not boxes_per_image:
        raise ValueError("No boxes found in dataset")
    
    boxes_per_image = np.array(boxes_per_image)
    
    # Define head receptive bands (stride 8, 16, 32 at 640px input)
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
    
    results = {
        "P3_stride_8": distribution_stats(*stride_8_band),
        "P4_stride_16": distribution_stats(*stride_16_band),
        "P5_stride_32": distribution_stats(*stride_32_band),
        "total_images": len(boxes_per_image),
        "total_boxes": len(boxes_per_image),
        "overall_mean_px": float(np.mean(boxes_per_image)),
        "global_min_px": float(np.min(boxes_per_image)),
        "global_max_px": float(np.max(boxes_per_image))
    }
    
    # Generate visualization
    plt.figure(figsize=(8, 6))
    labels = ["P3 (<64px)", "P4 (64-128px)", "P5 (>128px)"]
    shares = [results[f"P{i}_stride_{8*i}"]["share_percent"] for i in range(1, 4)]
    
    bars = plt.bar(labels, shares, color=['#3b82f6', '#8b5cf6', '#ec4899'])
    plt.ylabel("Share of Boxes (%)")
    plt.title("Box Scale Distribution by Detection Head")
    plt.ylim(0, max(shares) * 1.2)
    
    # Add percentage labels
    for bar, share in zip(bars, shares):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{share:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    output_path = Path(output_dir) / "box_scale_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plot to: {output_path}")
    print("\nBox Scale Analysis:")
    print(f"| Detection Head | Receptive Band | Share of Boxes |")
    print(f"|---|---|---|")
    for head in ["P3_stride_8", "P4_stride_16", "P5_stride_32"]:
        result = results[head]
        if head == "P3_stride_8":
            band = "< 64 px"
        elif head == "P4_stride_16":
            band = "64–128 px"
        else:
            band = "> 128 px"
        print(f"| {head} | {band} | {result['share_percent']:.1f}% |")
    
    return results


def recommend_head_removal(box_profile: dict) -> dict:
    """
    Analyze whether any detection head can be safely removed.
    
    Returns recommendation with confidence level and data-driven rationale.
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
    min_head = min(recommendations.items(), key=lambda x: x[1].get("confidence_rank", 0))
    all_mandatory = all(r["action"] == "retain" for r in recommendations.values())
    
    if all_mandatory:
        overall = {
            "can_remove_any_head": False,
            "recommendation": "Do NOT remove any detection head",
            "evidence": "Data shows load spread across all heads; dropping any would forfeit significant target coverage"
        }
    else:
        removable = [k for k, v in recommendations.items() if v["action"] == "consider_removing"]
        overall = {
            "can_remove_any_head": True,
            "candidate_heads": removable,
            "recommendation": f"Consider removing {' or '.join(removable)} after experimental validation",
            "risk": f"Removing these heads forfeits ~{sum(recommendations[h]['share_percent'] for h in removable):.1f}% of boxes"
        }
    
    recommendations["_overall"] = overall
    return recommendations


def check_augmentation_already_applied(dataset_path: str, config: dict) -> dict:
    """
    Detect if dataset already has augmentation baked in (e.g., Roboflow processed).
    
    Warning signs:
    - Very large train split vs small unique image count
    - Consistent high diversity metrics despite limited source images
    - Dataset provider notes baked-in transforms
    """
    warnings = []
    
    # Example: Check train/validation/test split ratios
    splits = config.get("splits", {})
    
    if splits.get("train_count", 0) > 2000:
        unique_images = config.get("unique_source_images", 1000)
        if unique_images and splits["train_count"] / unique_images > 2:
            warnings.append({
                "type": "possible_baked_augmentation",
                "severity": "high",
                "message": f"Train split is {splits['train_count']/unique_images:.1f}x larger than unique images; may have baked-in augmentation",
                "action": "Reduce custom augmentation aggression or disable by default transforms"
            })
    
    return {
        "warnings": warnings,
        "training_recommendation": "Use light augmentation profile" if warnings else "Standard augmentation acceptable"
    }
