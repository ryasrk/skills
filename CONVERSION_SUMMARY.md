# Skills Repository - Conversion Summary

## What Changed

All skills converted from **Python `.py` files** to **Markdown `.md` files** for agent readability and ease of use in OpenCode sessions.

### Old Structure (`.py` files)
```
Skills/category/subcategory/skill.py  # Python code to import/run
README.md                              # Documentation
```

**Problem:** Agents had to understand Python API, import paths, execute code, etc.

### New Structure (`.md` files)  
```
Skills/category/subcategory/skill.md  # Agent-friendly documentation
README.md                              # Category overview
```

**Benefit:** Agents can directly read and apply guidance without executing code.

---

## Converted Skills

| Skill | Format | Content |
|---|---|---|
| `experimental-design/constraints` | `.md` | Define hard deployment constraints before training |
| `experimental-design/benchmarking` | `.md` | Create reproducible benchmark harnesses with negative controls |
| `data-profiling/analysis` | `.md` | Profile data distributions before architectural changes |
| `weight-transfer/accounting` | `.md` | Track pretrained parameter inheritance rates |
| `conversion-analysis/boundary` | `.md` | Trace accuracy losses by export stage |
| `metrics-pipeline/generation` | `.md` | Store metrics as JSON, regenerate reports from code |
| `environment-testing/validation` | `.md` | Prove benchmarks aren't lying via negative controls |
| `limitations/documentation` | `.md` | Document structural constraints honestly |

---

## Each Skill.md Contains

Every skill file includes:

1. **🎯 When to Use** - Clear applicability criteria
2. **📋 Core Principles** - Key guiding rules
3. **🔧 Step-by-Step Instructions** - Actionable procedures with code examples
4. **💻 Example Command Sequences** - Copy-paste ready workflows
5. **⚠️ Common Mistakes** - What NOT to do
6. **✅ Success Indicators** - How you know it's done right
7. **🔗 Related Skills** - Where to look next
8. **📚 Reference Examples** - Real project cases from YOLOv11/OpenVINO
9. **🎯 Quick Checklist** - Immediate action items

Total: ~100+ actionable instructions across all skills.

---

## Repository Stats

- **Files added:** 8 new `.md` skill documents
- **Lines added:** ~2,995 lines of agent guidance
- **Categories:** 7 research skill areas
- **Sub-categories:** 8 specific implementations
- **Total scope:** Covers entire experimental ML lifecycle

---

## Why This Works Better for Agents

### Before (`skill.py`)
```python
def define_constraints(cpu_cores=2, memory_mb=2048):
    """Define constraints."""
    pass
```

Agent needs to:
1. Import module
2. Understand function signature  
3. Execute code
4. Parse output
5. Still figure out WHY this matters

### After (`skill.md`)
```markdown
### When to Use This Skill
Apply when starting ANY edge/deployment-constrained ML project...

### Step 1: Define Hardware Budget
```json
{
    "cpu_cores": 2,
    "memory_max_mb": 2048
}
```

### ⚠️ Common Mistake: Defining Constraints After Training
WRONG: Train freely, then ask "Can it fit?"
CORRECT: Define constraints FIRST
```

Agent can immediately:
1. Read principles
2. See exact format needed
3. Avoid common mistakes
4. Apply to their own project

No execution overhead, just direct application.

---

## Next Steps

For agents working on new ML projects:

1. **Read `USAGE.md`** first to understand workflow
2. **Start with constraint definition**: `experimental-design/constraints/skill.md`
3. **Before any architecture change**: `data-profiling/analysis/skill.md`  
4. **Track weight transfer**: `weight-transfer/accounting/skill.md`
5. **Prove benchmarks aren't lying**: `environment-testing/validation/skill.md`
6. **Generate honest limitations**: `limitations/documentation/skill.md`

Each skill builds on previous ones, creating rigorous experimental methodology.

---

## Repository URL

https://github.com/ryasrk/skills.git

All 8 skill files ready for immediate agent consumption.
