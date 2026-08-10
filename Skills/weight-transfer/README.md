# Weight Transfer Skills

Track how many pretrained parameters actually transfer when modifying architectures — parameter counts lie, weight inheritance doesn't.

## Skills in this Category

### accounting
Measure % of model's parameters that transfer from pretrained checkpoints after architecture modifications.

**When to use:** Any fine-tuning scenario with <10k unique training samples  
**Key insight:** Width scaling changes channel counts → most tensors can't transfer  
**Output:** Transfer rate table, warning if transfer drops below 80%

## Example Usage

```python
from Skills.weight_transfer.accounting import check_weight_transfer

# Check transfer rate for width-scaled variant
transfer = check_weight_transfer(
    model_variant_path="configs/models/yolo11n-face-v1.yaml",
    pretrained_checkpoint="pretrained/coco.pt"
)

print(f"Transfer rate: {transfer['transfer_rate_percent']}%")
if transfer['warning_threshold_exceeded']:
    print("⚠️ Architecture changes may hurt more than help on small datasets")
```

## Key Findings from Research

- **V0 (stock):** 89% weight transfer from COCO  
- **V1 (width scaled):** 24% weight transfer → 1.6 mAP points worse despite fewer params  
- **V2 (attention removed):** 89% transfer → same accuracy as V0, 10% smaller

**Takeaway:** Keep architecture close to stock when you need pretrained knowledge.
