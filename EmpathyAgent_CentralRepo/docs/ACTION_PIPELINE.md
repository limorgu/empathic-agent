# Empathetic Action — Minimal LoRA Pipeline (Train → Infer → Score)

**Assumes** your runnable code lives at:  
`experiments/EmpathyAgentReplication_ex1/Code`

```bash
cd experiments/EmpathyAgentReplication_ex1/Code
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.3.1 transformers==4.44.2 peft==0.13.2 datasets==2.20.0             accelerate==0.33.0 evaluate==0.4.2 tqdm jsonlines scikit-learn

# verify paths
python cfg_paths.py

# ensure mini SFT link (optional sanity run)
ln -sf OriginalPaperEmpathyAgent/dataset/sft_empathyagent_mini.jsonl ./sft_empathyagent_mini.jsonl

# train tiny LoRA (CPU OK)
python training_code/train_lora_min.py   --max_steps 20   --output_name lora_tinyllama_min   --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0   --train_file ./sft_empathyagent_mini.jsonl

# build paper prompts/refs
python training_code/make_test_prompts.py   --src OriginalPaperEmpathyAgent/dataset/testset_100.json   --out_dir OriginalPaperEmpathyAgent/dataset
ln -sf OriginalPaperEmpathyAgent/dataset/test_prompts.jsonl ./test_prompts.jsonl
ln -sf OriginalPaperEmpathyAgent/dataset/test_refs.jsonl     ./test_refs.jsonl

# inference (choose one)
# sampling (temp=0.7):
python training_code/infer_lora_min.py   --adapter_name lora_tinyllama_min   --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0   --prompts_file ./test_prompts.jsonl   --max_new_tokens 160 --temperature 0.7
# greedy (paper-style deterministic):
python training_code/infer_lora_min.py   --adapter_name lora_tinyllama_min   --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0   --prompts_file ./test_prompts.jsonl   --max_new_tokens 160   --greedy

# export + score
python training_code/export_for_scoring.py --adapter_name lora_tinyllama_min
python training_code/score_actions_min.py   --adapter_name lora_tinyllama_min   --references ./test_refs.jsonl

# (optional) score vs a single gold key
python training_code/score_actions_min.py   --adapter_name lora_tinyllama_min   --references OriginalPaperEmpathyAgent/dataset/test_refs_empathy_goal_nl.jsonl
```

**Outputs:** `experiments/EmpathyAgentReplication_ex1/output/lora_tinyllama_min/`  
(LoRA adapter, `inference.jsonl`, `predictions.csv`, `scores.csv`)
