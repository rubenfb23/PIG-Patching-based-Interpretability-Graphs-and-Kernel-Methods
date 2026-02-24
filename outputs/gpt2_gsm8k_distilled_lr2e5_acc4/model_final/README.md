---
language:
- en
license: mit
library_name: transformers
tags:
- gpt2
- distillation
- gsm8k
- chain-of-thought
- causal-lm
pipeline_tag: text-generation
base_model:
- gpt2
datasets:
- HAD653/gsm8k-cot-120b
---

# GPT-2 Distilled on GSM8K CoT 120B

This model is a distilled checkpoint based on `gpt2`, trained on data derived from the GSM8K CoT dataset:
- Source dataset: https://github.com/HAD653/gsm8k-cot-120b

## Checkpoint details

- Base model: `gpt2`
- Training artifact name: `gpt2_gsm8k_distilled_lr2e5_acc4/model_final`
- Intended use: research/experimentation on math reasoning style transfer and distillation behavior.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo_id = "rubenfb23/gpt2-gsm8k-cot-120b-distilled-lr2e5-acc4"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)
```

## Notes

- This repository currently contains the final distilled weights and tokenizer files from the local run.
- Evaluate outputs before production use.
