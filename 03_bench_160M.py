import torch
from pathlib import Path
from lm_eval import utils, simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# Benchmark Script for Pythia 160M models
# using the EleutherAI LM-Eval Harness

base_dir = "/vol/tmp/koppelmm"
base_path = Path(base_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
pythia_minipile = AutoModelForCausalLM.from_pretrained(base_path / "pythia160m_minipile_DensityProportionedHigh_trained", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(base_path / "pythia160m_dedup_untrained", use_fast=True, local_files_only=True) # Use exact same tokenizer
pythia_minipile = pythia_minipile.to(device)
 
batch_size_hflm = 1

pythia_minipile_hflm = HFLM(pretrained=pythia_minipile,
                        tokenizer=tokenizer,
                        batch_size=batch_size_hflm)

results = simple_evaluate(model=pythia_minipile_hflm,
                          tasks=["arc_challenge", "mmlu", "winogrande", "hellaswag", "lambada", "blimp"],
                          num_fewshot=0, # zero-shot
                          batch_size=batch_size_hflm,
                          device="cuda",
                          limit=None)

with open('04_eval_160M_minipile_DensityProportionedHigh.txt', 'w') as f:
    f.write(str(results))

# Make the table and save it too
table = utils.make_table(results)
print(table)
with open('04_eval_160M_minipile_DensityProportionedHigh_table.txt', 'w') as f:
    f.write(table)

# tmux new -s bench_160M
# tmux attach -t bench_160M
# conda activate minipile
# 
# CUDA_VISIBLE_DEVICES=1 python 03_bench_160M.py
# 
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t bench_160M
# tmux list-sessions
# tmux kill-session -t bench_160M