# Central Repo Structure

This repository is the **documentation + results hub**. Place your runnable code under `experiments/` so readers have a single entry point.

```
EmpathyAgent_CentralRepo/
  README.md
  RESULTS.md
  docs/
    ACTION_PIPELINE.md
    SCENARIO_PIPELINE.md
    REPO_STRUCTURE.md
  experiments/
    EmpathyAgentReplication_ex1/
      Code/
      data/          # (excluded by .gitignore)
      output/        # (excluded by .gitignore)
```

## Why separate central docs from experiments?

- Keeps the public repo lightweight and readable.
- Lets you iterate on experiments without cluttering documentation.
- Prevents large artifacts from ballooning the git history.

## What to commit vs ignore

- **Commit:** scripts, READMEs, configuration, small CSV summaries.
- **Ignore:** model weights, large outputs, raw datasets (already handled in `.gitignore`).
