# EmpathyAgent — Central Project Hub

A central, lightweight repo that **summarizes the project**, provides **clear, reproducible steps**, and links to concrete experiments under `experiments/`.

- ✅ **What’s inside**: docs, results, and a canonical README that anyone can follow
- 🧪 **Where to put code**: place your working experiment folder(s) under `experiments/` (see below)
- 🔗 **Upstream**: mirrors the EmpathyAgent paper tasks with a minimal LoRA pipeline

---

## Quick Start (for new readers)

```bash
git clone <THIS_REPO_URL>
cd EmpathyAgent_CentralRepo
# open docs:
# - docs/ACTION_PIPELINE.md (empathetic_action)
# - docs/SCENARIO_PIPELINE.md (scenario_understanding)
# - RESULTS.md (current numbers)
```

## Folder Layout

```
EmpathyAgent_CentralRepo/
  README.md                 # this file
  RESULTS.md                # current metrics (Action + Scenario)
  docs/
    ACTION_PIPELINE.md      # step-by-step: train → infer → score (action)
    SCENARIO_PIPELINE.md    # step-by-step: prompts → infer → BERTScore (scenario)
    REPO_STRUCTURE.md       # how experiments fit in and what to commit
  experiments/
    EmpathyAgentReplication_ex1/   # ← put your working folder here (Code/, data/, output/)
```

> Keep data/outputs out of git. The supplied `.gitignore` excludes large artifacts by default.

## How to add your experiment

1. Move or copy your existing working directory into `experiments/`, e.g.:
   ```bash
   mv ~/Documents/to_git/EmpathyAgentReplication/EmpathyAgentReplication_ex1       /path/to/EmpathyAgent_CentralRepo/experiments/
   ```
2. Commit the central repo and push:
   ```bash
   cd /path/to/EmpathyAgent_CentralRepo
   git init
   git add -A
   git commit -m "Central hub + docs + results"
   git branch -M main
   git remote add origin https://github.com/<you>/<EmpathyAgent_CentralRepo>.git
   git push -u origin main
   ```

## Where to look next

- **Action pipeline**: `docs/ACTION_PIPELINE.md`
- **Scenario pipeline**: `docs/SCENARIO_PIPELINE.md`
- **Latest results**: `RESULTS.md`
- **Your runnable code**: `experiments/EmpathyAgentReplication_ex1/Code/…` (once you place it here)
