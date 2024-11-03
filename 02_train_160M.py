# Related to 02_eval_160M.ipynb
# Training script for Distributed Training of Pythia 160M on MiniPile

import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

##
##
# torchrun --nproc_per_node=<<NUM GPUs>> 02_train_160M.py
##
##

def training():
    base_dir = "/mnt/data"
    base_path = Path(base_dir)

    # Loading minipile from the local directory 
    # https://stackoverflow.com/questions/77020278/how-to-load-a-huggingface-dataset-from-local-path
    # https://github.com/MK2112/mobileYOLOv3/blob/main/mobileyolov3-cocotext.ipynb
    # Split is named exactly like with the original dataset https://huggingface.co/datasets/JeanKaddour/minipile
    print('Loading MiniPile train + val datasets...')
    minipile_train = load_dataset("parquet",
                                data_files={
                                    "train": str(base_path / "MiniPile" / "data" / "train-*.parquet"),
                                    "validation": str(base_path / "MiniPile" / "data" / "validation-*.parquet"),
                                    "test": str(base_path / "MiniPile" / "data" / "test-*.parquet")
                                },
                                cache_dir=str(base_path / "MiniPile_Cache"),
                                split="train")

    minipile_val = load_dataset("parquet",
                                data_files={
                                    "train": str(base_path / "MiniPile" / "data" / "train-*.parquet"),
                                    "validation": str(base_path / "MiniPile" / "data" / "validation-*.parquet"),
                                    "test": str(base_path / "MiniPile" / "data" / "test-*.parquet")
                                },
                                cache_dir=str(base_path / "MiniPile_Cache"),
                                split="validation")

    # Load the untrained Pythia 160M tokenizer and model
    # https://stackoverflow.com/questions/64001128/load-a-pre-trained-model-from-disk-with-huggingface-transformers
    tokenizer = AutoTokenizer.from_pretrained(base_path / "pythia160m_dedup_untrained", use_fast=True, local_files_only=True)
    empty_model = AutoModelForCausalLM.from_pretrained(base_path / "pythia160m_dedup_untrained", local_files_only=True)

    def tokenize(example): # seq_len = max_length = 2048 (always)
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=2048)

    minipile_train_tokenized = minipile_train.map(tokenize, batched=True)
    minipile_train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"]) # new fields from tokenizing

    # Not really needed, but we have it, might as well make it serve as a reference for investigation of the model's performance
    minipile_val_tokenized = minipile_val.map(tokenize, batched=True)
    minipile_train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"]) # new fields from tokenizing

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.distributed.init_process_group("nccl")

    output_dir = str(base_path / "pythia160m_minipile_trained")
    log_dir = str(base_path / "160m_minipile_logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


    # https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1.5,            # Since train_iters gets set, use num_train_epochs=1.5 like for The Pile
        per_device_train_batch_size=8,   # Gives an effective batch size of 1024 after grad accum
        per_device_eval_batch_size=8,    # Same as training batch size
        gradient_accumulation_steps=128, # Achieve a batch size of 1024
        learning_rate=6e-4,              # Default Pythia 160M
        weight_decay=0.01,               # Default Pythia 160M
        max_steps=1024,                  # Adjusted for MiniPile
        lr_scheduler_type="cosine",      # As per Pythia 160M paper
        warmup_steps=int(0.01 * 1024),   # 1% of total steps for warmup
        logging_dir=log_dir,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,     # Frequency for evaluation during training
        save_steps=1024,    # Save at the end of training
        save_total_limit=1, # Only keep the most recent checkpoint
        fp16=False,         # Not using mixed precision for comparable conditions
        report_to="none",   # Noting this for later iterations, maybe set this as "wandb", "tensorboard" or smth
        ddp_find_unused_parameters=False, # see https://discuss.pytorch.org/t/how-to-change-ddp-parameter-find-unused-parameters-true-to-false-during-training/130763
    )

    # Train Pythia 160M Untrained on MiniPile
    # https://huggingface.co/docs/transformers/v4.46.0/en/main_classes/trainer
    trainer = Trainer(model=empty_model,
                      args=training_args,
                      train_dataset=minipile_train_tokenized,
                      eval_dataset=minipile_val_tokenized)

    trainer.train()
    trainer.save_model(str(base_path / "pythia160m_minipile_trained")) # This saves the model weights
    tokenizer.save_pretrained(str(base_path / "pythia160m_minipile_trained")) # This saves the tokenizer (don't know if needed, better save than sorry)

if __name__ == "__main__":
    training()