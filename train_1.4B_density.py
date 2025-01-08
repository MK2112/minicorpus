import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from torch.optim.lr_scheduler import _LRScheduler
from transformers import DataCollatorForLanguageModeling
from lm_eval import utils, simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, get_cosine_schedule_with_warmup

#####
## Load, Train, Bench, Push, Clean script for Pythia 1.4B on MiniPile Density-Proportioned
#####

###
# Don't place scripts of this type in the same folder or in child/parent folders of one another 
# The script deletes what it used from within the folder
# Therefore may interfere with other scripts of its type in this or child folders
###

# Everything will be done in the dir this script is placed into

# ! pip install torch (for GPU)
# ! pip install lm-eval
# ! pip install transformers datasets


## I ran with:
## CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 03_train_1.4B_density.py
## on 3x A100

base_dir = str(Path(__file__).parent.absolute())
base_path = Path(base_dir)

model_target = "pythia1.4b_dedup_untrained"
model_cache = "pythia1.4b_dedup_untrained_Cache"
model_repo = "EleutherAI/pythia-1.4b-deduped"
model_branch = "step0"

dataset_target = "MiniPile_Density"
dataset_cache = "MiniPile_Density_Cache"
dataset_repo = "Marcus2112/minipile_density-proportioned"

output_folder = "pythia1.4b_minipile_density_trained"
log_folder = "1.4b_minipile_density_logs" # This is will remain empty

bench_txt = str(base_path / output_folder / "03_eval_1.4B_minipile_density.txt")
bench_table_txt = str(base_path / output_folder / "03_eval_1.4B_minipile_density_table.txt")

batch_size = 2 # Thats the max for 3x A100

def download_model(down_dir: str, target_folder: str, cache_folder: str, repo_id: str, branch: str = "main") -> None:
    down_dir = Path(down_dir)
    target_dir = down_dir / target_folder
    cache_dir = down_dir / cache_folder
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {repo_id}/{branch}...")
    while True:
        try:
            snapshot_download(repo_id,
                              repo_type="model",
                              revision=branch,
                              cache_dir=str(cache_dir),
                              local_dir=str(target_dir))
            break
        except Exception as e:
            print(f"Download attempt failed: {e}")
            continue

def download_dataset(down_dir: str, target_folder: str, cache_folder: str, repo_id: str) -> None:
    down_dir = Path(down_dir)
    target_dir = down_dir / target_folder
    cache_dir = down_dir / cache_folder
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {repo_id}...")
    # I tried fiddling with the os.environs, because I wanted to use the load_dataset function, but that is stubborn
    # we actually don't need that, snapshot_download suffices
    while True:
        try:
            snapshot_download(repo_id, repo_type="dataset", cache_dir=str(cache_dir), local_dir=str(target_dir))
            break
        except Exception as _:
            continue

class CosineSchedulerWithMinLR(_LRScheduler):
    # Basically wrapping the get_cosing_schedule_with_warmup in a lower-bound setting
    # Allows for Cosine Scheduling with a min_lr enforced
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr):
        self.min_lr = min_lr
        self.base_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                              num_warmup_steps=num_warmup_steps,
                                                              num_training_steps=num_training_steps)
        super().__init__(optimizer)
        
    def get_lr(self):
        lrs = self.base_scheduler.get_lr()
        return [max(lr, self.min_lr) for lr in lrs]
        
    def step(self):
        self.base_scheduler.step()
        super().step() 

def training():
    minipile_train = load_dataset("parquet",
                                  data_files={"train": str(base_path / dataset_target / "data/train-*.parquet")},
                                  cache_dir=str(base_path / dataset_cache),
                                  split="train")

    minipile_val = load_dataset("parquet",
                                data_files={"validation": str(base_path / dataset_target / "data/validation-*.parquet")},
                                cache_dir=str(base_path / dataset_cache),
                                split="validation")

    tokenizer = AutoTokenizer.from_pretrained(base_path / model_target, use_fast=True, local_files_only=True)
    empty_model = AutoModelForCausalLM.from_pretrained(base_path / model_target, local_files_only=True, low_cpu_mem_usage=True)

    # Tokenizer doesn't have a pad token, use EOS as a substitute
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize(example): 
        # seq_len = max_length = 2048 (as upper boundary, so not strict size -> no padding needed)
        return tokenizer(example["text"], 
                         truncation=True, 
                         max_length=2048,
                         return_special_tokens_mask=True)

    # I deleted old iterations of this folder before running this script for first time
    if os.path.exists(base_path / f"{dataset_target}_train_tokenized"):
        minipile_train_tokenized = load_dataset("arrow", data_files=str(base_path / f"{dataset_target}_train_tokenized/*.arrow"), split="train")
        minipile_val_tokenized = load_dataset("arrow", data_files=str(base_path / f"{dataset_target}_val_tokenized/*.arrow"), split="train")
    else:
        minipile_train_tokenized = minipile_train.map(tokenize, batched=True, remove_columns=minipile_train.column_names) # retain only new fields from tokenization
        minipile_val_tokenized = minipile_val.map(tokenize, batched=True, remove_columns=minipile_val.column_names)
        minipile_train_tokenized.save_to_disk(base_path / f"{dataset_target}_train_tokenized")
        minipile_val_tokenized.save_to_disk(base_path / f"{dataset_target}_val_tokenized")

    total_batch = 1024

    # Dynamic padding during training (mlm -> mask language model -> we're doing causal here)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device_count = torch.cuda.device_count()
        print(f"Available GPUs: {device_count}")
        # Just because I can
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            device_properties = torch.cuda.get_device_properties(device)
            total_mem = device_properties.total_memory / (1024 ** 3)
            allocd_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)
            free_mem = total_mem - allocd_mem
            print(f"\nGPU {i}:\t{device_properties.name}")
            print(f"\tTotal memory:\t\t{total_mem:.2f} GiB")
            print(f"\tAllocated memory:\t{allocd_mem:5.2f} GiB")
            print(f"\tFree memory:\t\t{free_mem:.2f} GiB")
    else:
        print("No CUDA-capable GPUs available")

    output_dir = str(base_path / output_folder)
    log_dir = str(base_path / log_folder)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1.5,            # Since train_iters gets set, use num_train_epochs=1.5 like for The Pile
        per_device_train_batch_size=batch_size,   # Gives an effective batch size of 1024 after grad accum
        per_device_eval_batch_size=batch_size,    # Same as training batch size
        gradient_accumulation_steps=(total_batch // batch_size // device_count), # Achieve a batch size of 1024
        learning_rate=2e-4,              # Default Pythia 1.4B-specific
        weight_decay=0.01,               # Default Pythia 160M and 1.4B
        max_steps=1024,                  # Adjusted for MiniPile (https://discuss.huggingface.co/t/how-does-max-steps-affect-the-number-of-samples-the-model-sees/69681)
        warmup_steps=int(0.01 * 1024),   # 1% of total steps for warmup
        logging_dir=log_dir,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,     # Frequency for evaluation during training
        save_steps=1024,    # Save at the end of training
        save_total_limit=1, # Only keep the most recent checkpoint
        fp16=True,          # Using mixed precision for comparable conditions
        report_to=None,     # Noting this for later iterations, maybe set this as "wandb", "tensorboard" or smth
        ddp_find_unused_parameters=False, # see https://discuss.pytorch.org/t/how-to-change-ddp-parameter-find-unused-parameters-true-to-false-during-training/130763
        max_grad_norm=1.0,  # Default Pythia 160M and 1.4B
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        #gradient_checkpointing=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    empty_model = empty_model.to(device)

    optimizer = Adam(empty_model.parameters(), lr=training_args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    scheduler = CosineSchedulerWithMinLR(optimizer=optimizer,
                                         num_warmup_steps=training_args.warmup_steps,
                                         num_training_steps=training_args.max_steps,
                                         min_lr=2e-5) # Pythia 1.4B-specific

    # https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer
    trainer = Trainer(model=empty_model,
                      args=training_args,
                      train_dataset=minipile_train_tokenized,
                      eval_dataset=minipile_val_tokenized,
                      data_collator=data_collator,
                      optimizers=(optimizer, scheduler))

    trainer.train()
    trainer.save_model(str(base_path / output_folder)) # This saves the model weights

def bench():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pythia_pile = AutoModelForCausalLM.from_pretrained(base_path / output_folder, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path / model_target, use_fast=True, local_files_only=True)
    pythia_pile = pythia_pile.to(device)
    
    batch_size_hflm = 1

    pythia_minipile_hflm = HFLM(pretrained=pythia_pile,
                                tokenizer=tokenizer,
                                batch_size=batch_size_hflm)

    results = simple_evaluate(model=pythia_minipile_hflm,
                              tasks=["arc_challenge", "mmlu", "winogrande", "hellaswag", "lambada", "blimp", "arc_easy"],
                              num_fewshot=0,
                              batch_size=batch_size_hflm,
                              device="cuda",
                              limit=None)

    # Packaging benchmark results into model folder to have them uploaded to HFHub too
    with open(bench_txt, 'w') as f:
        f.write(str(results))

    # Make the table and save it too
    table = utils.make_table(results)
    print(table)
    with open(bench_table_txt, 'w') as f:
        f.write(table)

def push_model():
    api = HfApi(token="xyz") # This is a temporary account, just for these scripts
    local_path = str(base_path / output_folder)
    hf_repo_id = "Marcus1221/pythia-1.4b-minipile_density-proportioned" # This is a temporary account, just for these scripts
    print(f"Pushing model to {hf_repo_id}")
    api.create_repo(hf_repo_id, exist_ok=True)
    api.upload_folder(folder_path=local_path,
                      repo_id=hf_repo_id,
                      repo_type="model")

def file_cleanup():
    print("Starting cleanup process...")
    for folder in [model_cache, dataset_cache, f"{dataset_target}_train_tokenized", f"{dataset_target}_val_tokenized", log_folder]:
        path = base_path / folder
        if path.exists():
            shutil.rmtree(path)
        else:
            print(f"Skipping deletion of {folder}")
    for file in [bench_txt, bench_table_txt]:
        path = Path(file)
        if path.exists():
            path.unlink()
        else:
            print(f"Skipping delection of {file}")    
    print("Concluded Cleanup.")

if __name__ == "__main__":
    # Load it, Train it, Bench it, Push it, Clean it
    download_model(down_dir=base_dir,
                   target_folder=model_target, 
                   cache_folder=model_cache,
                   repo_id=model_repo,
                   branch=model_branch)
    download_dataset(down_dir=base_dir,
                     target_folder=dataset_target, 
                     cache_folder=dataset_cache,
                     repo_id=dataset_repo)
    training()
    torch.cuda.empty_cache()
    bench()
    torch.cuda.empty_cache()
    push_model()
    file_cleanup()