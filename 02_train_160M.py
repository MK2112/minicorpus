# Adapted from 02_eval_160M.ipynb
# Training script for Distributed Training of Pythia 160M on MiniPile

import gc
import os
import torch
import numpy as np

from pathlib import Path
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling)

seed = 42

# This script expects datasets and models to be stored offline already
# See 02_eval_160M.ipynb for details on how to download and prepare the datasets and models

def training():
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # GPU ID
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # Number of GPUs
    
    base_dir = "/mnt/data"
    base_path = Path(base_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(local_rank)

    if local_rank == 0:
        print('Loading MiniPile train + val datasets...')
    
    # Load original tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path / "pythia160m_dedup_untrained",
                                              use_fast=True,
                                              local_files_only=True,
                                              model_max_length=2048) # maximum len for inputs to the model

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if not os.path.exists(base_path / "minipile_train_tokenized"):
        minipile_train = load_dataset("parquet",
                                      data_files={
                                          "train": str(base_path / "MiniPile" / "data" / "train-*.parquet"),
                                      },
                                      split="train")

        minipile_val = load_dataset("parquet",
                                    data_files={
                                        "validation": str(base_path / "MiniPile" / "data" / "validation-*.parquet"),
                                    },
                                    split="validation")

        def tokenize(example):
            return tokenizer(example["text"],
                             truncation=True,
                             max_length=2048,
                             padding=False,
                             return_special_tokens_mask=True)

        # Tokenize datasets
        minipile_train_tokenized = minipile_train.map(tokenize, batched=True, remove_columns=minipile_train.column_names, num_proc=1)
        minipile_val_tokenized = minipile_val.map(tokenize, batched=True, remove_columns=minipile_val.column_names, num_proc=1)

        # Save tokenized datasets
        if local_rank == 0:
            minipile_train_tokenized.save_to_disk(base_path / "minipile_train_tokenized")
            minipile_val_tokenized.save_to_disk(base_path / "minipile_val_tokenized")

    minipile_train_tokenized = load_dataset("arrow", 
                                            data_files=str(base_path / "minipile_train_tokenized/*.arrow"),
                                            streaming=True,
                                            split="train").with_format("torch")
    minipile_val_tokenized = load_dataset("arrow", 
                                          data_files=str(base_path / "minipile_val_tokenized/*.arrow"),
                                          streaming=True,
                                          split="train").with_format("torch")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Load model
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(base_path / "pythia160m_dedup_untrained", local_files_only=True, low_cpu_mem_usage=True)
    model = model.to(device)

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # Calculate Batch Distribution Details
    effective_batch_size = 1024
    per_gpu_batch_size = 2
    gradient_accumulation_steps = effective_batch_size // (per_gpu_batch_size * world_size)

    output_dir = str(base_path / "pythia160m_minipile_trained")
    log_dir = str(base_path / "160m_minipile_logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1.5,  # Ideally this gets overriden by max_steps, but just in case
        per_device_train_batch_size=per_gpu_batch_size,
        per_device_eval_batch_size=per_gpu_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=6e-4,
        weight_decay=0.01,
        max_steps=1024,
        lr_scheduler_type="cosine",
        warmup_steps=int(0.01 * 1024),
        logging_dir=log_dir,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=1024,
        save_total_limit=1,
        fp16=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
    )

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=minipile_train_tokenized,
                      eval_dataset=minipile_val_tokenized,
                      data_collator=data_collator)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    trainer.train()

    # Save model only on main process
    if local_rank == 0:
        trainer.save_model(str(base_path / "pythia160m_minipile_trained"))
        tokenizer.save_pretrained(str(base_path / "pythia160m_minipile_trained"))

##
# torchrun --nproc_per_node=<<NUM GPUs>> 02_train_160M.py
##

if __name__ == "__main__":
    # Print GPU details
    if torch.cuda.is_available():
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
    # Start training
    training()

# tmux new -s 160m_minipile
# conda activate minipile
# torchrun --nproc_per_node=1 02_train_160M.py
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t 160m_minipile
# tmux list-sessions
# tmux kill-session -t <session_name>
