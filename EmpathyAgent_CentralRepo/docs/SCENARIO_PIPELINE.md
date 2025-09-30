# Scenario Understanding — Inference → BERTScore

**Assumes** your runnable code lives at:  
`experiments/EmpathyAgentReplication_ex1/Code`

```bash
cd experiments/EmpathyAgentReplication_ex1/Code
source .venv/bin/activate

# build scenario prompts from action prompts
python - <<'PY'
from pathlib import Path
import json
src = Path('OriginalPaperEmpathyAgent/dataset/test_prompts.jsonl')
dst = Path('OriginalPaperEmpathyAgent/dataset/scenario_prompts.jsonl')
with src.open('r', encoding='utf-8', errors='ignore') as f,      dst.open('w', encoding='utf-8') as wf:
    for line in f:
        if not line.strip(): continue
        row = json.loads(line)
        row['instruction'] = 'Summarize the situation in 1–2 sentences focusing on the main goal and constraints.'
        wf.write(json.dumps(row, ensure_ascii=False)+'\n')
print('✅ wrote', dst)
PY

# build scenario refs
python - <<'PY'
from pathlib import Path
import json
src = Path('OriginalPaperEmpathyAgent/dataset/testset_100.json')
dst = Path('OriginalPaperEmpathyAgent/dataset/scenario_refs.jsonl')
data = json.loads(src.read_text(encoding='utf-8', errors='ignore'))
if isinstance(data, dict):
    for v in data.values():
        if isinstance(v, list):
            data = v; break
with dst.open('w', encoding='utf-8') as f:
    for i, ex in enumerate(data):
        ref = ex.get('high_level_plan','')
        if isinstance(ref, (list, dict)):
            ref = json.dumps(ref, ensure_ascii=False)
        f.write(json.dumps({'id': i, 'reference': ref or ''}, ensure_ascii=False)+'\n')
print('✅ wrote', dst)
PY

# inference on scenario prompts
python training_code/infer_lora_min.py   --adapter_name lora_tinyllama_min   --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0   --prompts_file OriginalPaperEmpathyAgent/dataset/scenario_prompts.jsonl   --max_new_tokens 120 --temperature 0.6

# export predictions
python training_code/export_for_scoring.py --adapter_name lora_tinyllama_min

# score with BERTScore (CPU-friendly)
pip install bert-score
python training_code/score_scenario_bertscore.py   --adapter_name lora_tinyllama_min   --references OriginalPaperEmpathyAgent/dataset/scenario_refs.jsonl   --model_type distilroberta-base
```
