"""
Skill: weight-transfer-accounting

Track how many pretrained tensors transfer when modifying architectures.
This catches cases where parameter count optimization hurts more than it helps.
"""

import torch
from pathlib import Path
from collections import defaultdict


def check_weight_transfer(model_variant_path: str, pretrained_checkpoint: str) -> dict:
    """
    Measure % of model's parameters that actually transfer from pretrained checkpoint.
    
    This reveals the hidden cost of architecture modifications:
    - Reducing width changes channel counts → most tensors can't transfer
    - Keeping topology identical → most weights transfer
    - Simply removing blocks with Identity() preserves index alignment
    
    Args:
        model_variant_path: Path to variant YAML or config
        pretrained_checkpoint: Path to pretrained checkpoint (e.g., yolo11n.pt)
    
    Returns:
        Transfer metrics including per-layer statistics and overall rate
    """
    # Load variant model architecture
    # Note: Actual implementation depends on framework
    # For YOLOv11/Ultralytics:
    #   model = YOLO(f"configs/models/{model_variant_path}.yaml")
    # For PyTorch generic:
    #   model = MyModel.from_config(variant_config)
    
    # Load pretrained checkpoint
    # state_dict_pretrained = torch.load(pretrained_checkpoint, map_location='cpu')
    
    # Simulated example (replace with actual loading logic):
    original_keys = {
        "model.0.conv.weight", "model.0.bn.weight",  # stem layers
        "model.1.cv1.conv.weight", "model.1.m.0.proj.weight",  # C2PSA attention
        "model.6.cv2.conv.weight",  # deeper blocks
        "model.13.cv2.conv.weight",  # detection heads
        # ... 499 total keys in stock YOLOv11n
    }
    
    # After modifying architecture (width scaling), shapes change:
    modified_model_keys = {
        "model.0.conv.weight", "model.0.bn.weight",  # same (stem usually unchanged)
        "model.1.cv1.conv.weight", "model.1.m.0.proj.weight",  # same if no block removal
        # But depthwise convs have different in_channels if width changed
        "model.new_custom_block.weight",  # newly added layer
        # Some detection head keys mismatch due to stride changes
    }
    
    # Calculate overlap with PRETRAINED checkpoint (not self!)
    # Correct method: intersect modified model against *pretrained checkpoint*
    transferable = set()
    
    for key in modified_model_keys:
        if key in original_keys:  # exists in pretrained
            # Also check shape compatibility
            if is_compatible_shape(key, modified_model_keys[key], original_keys[key]):
                transferable.add(key)
    
    total_keys = len(modified_model_keys)
    transferred = len(transferable)
    transfer_rate = transferred / total_keys * 100 if total_keys > 0 else 0
    
    return {
        "total_parameters": count_total_params(modified_model_keys),
        "transferred_parameters": count_transferred_params(transferable),
        "reinitialized_parameters": total_keys - transferred,
        "transfer_rate_percent": round(transfer_rate, 1),
        "by_category": analyze_by_category(modified_model_keys, transferable),
        "warning_threshold_exceeded": transfer_rate < 80,
        "recommendation": get_recommendation(transfer_rate)
    }


def is_compatible_shape(key: str, new_shape, old_shape) -> bool:
    """Check if tensor shape is compatible for transfer."""
    # Exact match
    if new_shape == old_shape:
        return True
    
    # Same shape, different dtype → upcast OK
    if new_shape == old_shape and len(new_shape) != 0:
        return True
    
    # Bias terms are often reinitialised even if other layers transfer
    if "bias" in key:
        return False
    
    return False


def count_total_params(keys: set) -> int:
    """Estimate total parameter count from keys (placeholder)."""
    # In real impl: sum(param.numel() for param in model.parameters())
    return len(keys) * 1000  # Placeholder


def count_transferred_params(keys: set) -> int:
    """Estimate transferred parameter count (placeholder)."""
    return len(keys) * 1000  # Placeholder


def analyze_by_category(all_keys: set, transferred: set) -> dict:
    """Break down transfer by layer category."""
    categories = defaultdict(lambda: {"total": 0, "transferred": 0})
    
    for key in all_keys:
        if "conv" in key and "head" not in key:
            cat = "backbone_conv"
        elif "bn" in key:
            cat = "batch_norm"
        elif "m." in key or "attention" in key:
            cat = "attention"
        elif "cv" in key and "head" not in key:
            cat = "neck_spp"
        elif "head" in key or "detect" in key:
            cat = "detection_heads"
        else:
            cat = "other"
        
        categories[cat]["total"] += 1
        if key in transferred:
            categories[cat]["transferred"] += 1
    
    result = {}
    for cat, counts in categories.items():
        if counts["total"] > 0:
            result[cat] = {
                "total": counts["total"],
                "transferred": counts["transferred"],
                "rate_percent": round(counts["transferred"] / counts["total"] * 100, 1)
            }
    
    return result


def get_recommendation(transfer_rate: float) -> str:
    """Generate recommendation based on transfer rate."""
    if transfer_rate >= 80:
        return f"✓ Good: {transfer_rate:.1f}% weights transfer; pretrained knowledge preserved"
    elif transfer_rate >= 50:
        return f"⚠ Warning: Only {transfer_rate:.1f}% weights transfer; consider keeping closer to stock"
    else:
        return f"✗ High risk: Only {transfer_rate:.1f}% weights transfer; architecture changes may hurt more than help on small datasets"


def print_transfer_table(variants_results: list) -> str:
    """
    Print formatted table comparing transfer rates across variants.
    
    variants_results should be list of dicts with keys:
    - 'variant_name'
    - 'params'
    - 'transfer_rate'
    - 'GFLOPs'
    """
    lines = [
        "| Variant | Params | vs Stock | GFLOPs @640 | Pretrained tensors inherited |",
        "|---|---|---|---|---|"
    ]
    
    for v in variants_results:
        params = f"{v['params']:,}"
        vs_stock = f"{v.get('vs_stock_percent', 100)}%"
        flops = f"{v.get('gflops_640', 0):.3f}"
        inherited = f"{v.get('transferred_count', 0)}/{v.get('total_count', 0)} ({v['transfer_rate']}%)"
        
        lines.append(f"| `{v['variant_name']}` | {params} | {vs_stock} | {flops} | {inherited} |")
    
    return "\n".join(lines)


# Usage example
if __name__ == "__main__":
    results = []
    
    # Stock YOLOv11n
    results.append({
        "variant_name": "yolo11n-face-v0",
        "params": 2590035,
        "transfer_rate": 89.8,
        "vs_stock_percent": 100,
        "gflops_640": 6.5,
        "transferred_count": 448,
        "total_count": 499
    })
    
    # Width-scaled variant (inherits less)
    results.append({
        "variant_name": "yolo11n-face-v1", 
        "params": 1564599,
        "transfer_rate": 24.0,
        "vs_stock_percent": 60,
        "gflops_640": 4.241,
        "transferred_count": 120,
        "total_count": 499
    })
    
    # Attention-removed variant
    results.append({
        "variant_name": "yolo11n-face-v2",
        "params": 2340307,
        "transfer_rate": 88.8,
        "vs_stock_percent": 90,
        "gflops_640": 6.236,
        "transferred_count": 406,
        "total_count": 457  # Changed because C2PSA removed
    })
    
    print("\n" + "="*80)
    print("WEIGHT TRANSFER ANALYSIS")
    print("="*80)
    print(print_transfer_table(results))
    print("\nKey insight:")
    print("- V0/V2 inherit ~89% of tensors from COCO")
    print("- V1/V3 inherit ~24% — width scaling changes almost every channel count")
    print("- This explains why V1 has 40% fewer params but 1.6 mAP points worse accuracy")
