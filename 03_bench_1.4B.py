import torch
from pathlib import Path
from lm_eval import utils, simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

base_dir = "/vol/tmp/koppelmm"
base_path = Path(base_dir)

# Benchmark Script for Pythia 1.4B models
# using the EleutherAI LM-Eval Harness
# Benchmarks conducted on "arc_challenge", "mmlu", "winogrande", "hellaswag", "lambada", "blimp", "arc_easy"

device = "cuda" if torch.cuda.is_available() else "cpu"
pythia_pile = AutoModelForCausalLM.from_pretrained(base_path / "pythia1.4b_dedup_pile", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(base_path / "pythia1.4b_dedup_pile", use_fast=True, local_files_only=True)
pythia_pile = pythia_pile.to(device)
 
batch_size_hflm = 1

pythia_minipile_hflm = HFLM(pretrained=pythia_pile,
                        tokenizer=tokenizer,
                        batch_size=batch_size_hflm)

results = simple_evaluate(model=pythia_minipile_hflm,
                          tasks=["arc_challenge", "mmlu", "winogrande", "hellaswag", "lambada", "blimp", "arc_easy"],
                          num_fewshot=0, # zero-shot
                          batch_size=batch_size_hflm,
                          device="cuda",
                          limit=None)

with open('04_eval_1.4B_dedup_pile_easy.txt', 'w') as f:
    f.write(str(results))

# Make the table and save it too
table = utils.make_table(results)
print(table)
with open('04_eval_1.4B_dedup_pile_table_easy.txt', 'w') as f:
    f.write(table)

# tmux new -s bench_14B
# tmux attach -t bench_14B
# conda activate minipile
# 
# CUDA_VISIBLE_DEVICES=2 python 03_bench_1.4B.py
#  
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t bench_14B
# tmux list-sessions
# tmux kill-session -t bench_14B