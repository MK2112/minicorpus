# Adapted from 02_eval_160M.ipynb
# Training script for Distributed Training of Pythia 160M on MiniPile

import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, get_scheduler

base_dir = "/mnt/data"
base_path = Path(base_dir)

def download_model(down_dir: str, target_folder: str, cache_folder: str, repo_id: str, branch: str = "main") -> None:
    down_dir = Path(down_dir)
    target_dir = down_dir / target_folder
    cache_dir = down_dir / cache_folder

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading {repo_id}/{branch}...")

    while True:
        try:
            snapshot_download(
                repo_id,
                repo_type="model",
                revision=branch,
                cache_dir=str(cache_dir),
                local_dir=str(target_dir)
            )
            break
        except Exception as e:
            print(f"Download attempt failed: {e}")
            continue

def training():
    download_model(down_dir=base_dir, target_folder="pythia160m_dedup_untrained", 
                   cache_folder="pythia160m_dedup_untrained_Cache",
                   repo_id="EleutherAI/pythia-160m-deduped", branch="step0")

    download_model(down_dir=base_dir, target_folder="pythia160m_dedup_pile", 
                   cache_folder="pythia160m_dedup_pile_Cache",
                   repo_id="EleutherAI/pythia-160m-deduped", branch="main")

    minipile_train = load_dataset("parquet",
                                  data_files={
                                      "train": str(base_path / "MiniPile" / "data" / "train-*.parquet"),
                                  },
                                  cache_dir=str(base_path / "MiniPile_Cache"),
                                  split="train")

    minipile_val = load_dataset("parquet",
                                data_files={
                                    "validation": str(base_path / "MiniPile" / "data" / "validation-*.parquet"),
                                },
                                cache_dir=str(base_path / "MiniPile_Cache"),
                                split="validation")

    tokenizer = AutoTokenizer.from_pretrained(base_path / "pythia160m_dedup_untrained", use_fast=True, local_files_only=True)
    empty_model = AutoModelForCausalLM.from_pretrained(base_path / "pythia160m_dedup_untrained", local_files_only=True, low_cpu_mem_usage=True)

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

    if os.path.exists(base_path / "minipile_train_tokenized"):
        minipile_train_tokenized = load_dataset("arrow", data_files=str(base_path / "minipile_train_tokenized/*.arrow"), split="train")
        minipile_val_tokenized = load_dataset("arrow", data_files=str(base_path / "minipile_val_tokenized/*.arrow"), split="train")
    else:
        minipile_train_tokenized = minipile_train.map(tokenize, batched=True, remove_columns=minipile_train.column_names) # retain only new fields from tokenization
        minipile_val_tokenized = minipile_val.map(tokenize, batched=True, remove_columns=minipile_val.column_names)
        minipile_train_tokenized.save_to_disk(base_path / "minipile_train_tokenized")
        minipile_val_tokenized.save_to_disk(base_path / "minipile_val_tokenized")

    batch_size = 2
    total_batch = 1024

    # Dynamic padding during training (mlm -> mask language model -> we're doing causal here)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device_count = torch.cuda.device_count()
        print(f"Available GPUs: {device_count}")
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

    output_dir = str(base_path / "pythia160m_minipile_trained")
    log_dir = str(base_path / "160m_minipile_logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1.5,            # Since train_iters gets set, use num_train_epochs=1.5 like for The Pile
        per_device_train_batch_size=4,   # Gives an effective batch size of 1024 after grad accum
        per_device_eval_batch_size=4,    # Same as training batch size
        gradient_accumulation_steps=256, # Achieve a batch size of 1024
        learning_rate=6e-4,              # Default Pythia 160M
        weight_decay=0.01,               # Default Pythia 160M
        max_steps=1024,                  # Adjusted for MiniPile (https://discuss.huggingface.co/t/how-does-max-steps-affect-the-number-of-samples-the-model-sees/69681)
        lr_scheduler_type="cosine",      # As per Pythia 160M paper
        warmup_steps=int(0.01 * 1024),   # 1% of total steps for warmup
        logging_dir=log_dir,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,     # Frequency for evaluation during training
        save_steps=1024,    # Save at the end of training
        save_total_limit=1, # Only keep the most recent checkpoint
        fp16=False,         # Not using mixed precision for comparable conditions
        report_to=None,     # Noting this for later iterations, maybe set this as "wandb", "tensorboard" or smth
        ddp_find_unused_parameters=False, # see https://discuss.pytorch.org/t/how-to-change-ddp-parameter-find-unused-parameters-true-to-false-during-training/130763
        max_grad_norm=1.0,  # As per Pythia 160M paper
    )

    # Ensure training across multiple GPUs if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    empty_model = empty_model.to(device)

    optimizer = Adam(empty_model.parameters(), lr=training_args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    # Train Pythia 160M Untrained on MiniPile
    # https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer
    trainer = Trainer(model=empty_model,
                    args=training_args,
                    train_dataset=minipile_train_tokenized,
                    eval_dataset=minipile_val_tokenized,
                    data_collator=data_collator,
                    optimizers=(optimizer, None))

    scheduler = get_scheduler(name=training_args.lr_scheduler_type,
                            optimizer=optimizer,
                            num_warmup_steps=training_args.warmup_steps,
                            num_training_steps=training_args.max_steps)

    num_batches = len(trainer.get_train_dataloader())  # Number of batches
    total_training_steps = num_batches * training_args.gradient_accumulation_steps * int(training_args.num_train_epochs)

    # Training loop with manual minimum learning rate enforcement
    for epoch in range(int(training_args.num_train_epochs)):
        with tqdm(total=total_training_steps, desc=f"Training Epoch {epoch + 1}/{int(training_args.num_train_epochs)}") as pbar:
            for _, batch in enumerate(trainer.get_train_dataloader()):
                trainer.training_step(trainer.model, batch)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                for param_group in optimizer.param_groups:
                    # Manually ... ensure lr doesn't go below min_lr (Pythia wants this)
                    param_group['lr'] = max(param_group['lr'], 6e-5)
                pbar.update(1)

    # Why is this a two-step process?!
    trainer.save_model(str(base_path / "pythia160m_minipile_trained")) # This saves the model weights
    tokenizer.save_pretrained(str(base_path / "pythia160m_minipile_trained")) # This saves the tokenizer (don't know if needed, better save than sorry)

if __name__ == "__main__":
    training()

# tmux new -s 160m_minipile
# conda activate minipile
# torchrun --nproc_per_node=1 02_train_160M.py
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t 160m_minipile
# tmux list-sessions
# tmux kill-session -t <session_name>
