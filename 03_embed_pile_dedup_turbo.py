import os
import gc
import torch
import queue
import threading
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

# Use this instead of 03_embed_pile_dedup.py

@dataclass
class Config:
    base_dir: str = "/vol/tmp/koppelmm"
    tokenization_batch_size: int = 128 # That's the max on 4x A6000
    prefetch_batches: int = 2
    embedding_dim: int = 768  # Doesn't do anything, but signals use of e5-base-4k
    shard_size: int = tokenization_batch_size * 4096  # Embeddings per shard (shards 0-51 exclusive, they were * 8192)
    num_worker_threads: int = 4
    max_length: int = 1024  # Contained at 1024 for better depth than e5-large but better speed than 4k

class AsyncWriter:
    def __init__(self, output_dir: Path, config: Config):
        self.output_dir = output_dir
        self.config = config
        self.write_queue = queue.Queue(maxsize=4)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        self.current_shard = self._get_next_shard_index()

    def _get_next_shard_index(self):
        # Count existing parquet files, determine the follow-up shard index
        existing_shards = list(self.output_dir.glob("shard_*.parquet"))
        return int(len(existing_shards))
        
    def _writer_loop(self):
        while True:
            try:
                data = self.write_queue.get()
                if data is None:
                    break
                    
                embeddings, texts = data
                table = pa.Table.from_arrays([pa.array(embeddings), pa.array(texts)], names=['embedding', 'text'])
                
                shard_path = self.output_dir / f"shard_{self.current_shard:09d}.parquet"
                pq.write_table(table, str(shard_path))
                self.current_shard += 1

                # For Memory Efficiency
                del embeddings, texts, table
            finally:
                self.write_queue.task_done()
                
    def write(self, embeddings: List[np.ndarray], texts: List[str]):
        self.write_queue.put((embeddings, texts))
        
    def finish(self):
        self.write_queue.put(None)
        self.writer_thread.join()

class TokenizationWorker:
    def __init__(self, tokenizer, config: Config, input_queue: queue.Queue, output_queue: queue.Queue):
        self.tokenizer = tokenizer
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while True:
            batch = self.input_queue.get()
            if batch is None:
                self.output_queue.put(None)
                break
            
            prefixed_texts = ["query: " + text for text in batch['text']]
            tokenized = self.tokenizer(prefixed_texts, max_length=self.config.max_length, 
                                       padding="max_length", truncation=True, return_tensors='pt')
            self.output_queue.put((tokenized, batch['text']))
            del prefixed_texts

class EmbeddingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.base_path = Path(config.base_dir)
        self.accelerator = Accelerator()
        self.setup_directories()
        
    def setup_directories(self):
        self.embd_dir = self.base_path / "Pile_Deduplicated_Embd"
        self.embd_dir.mkdir(exist_ok=True)
        
    def download_model(self):
        target_dir = self.base_path / "e5-base-4k"
        cache_dir = self.base_path / "e5-base-4k_Cache"
        
        for dir_path in [target_dir, cache_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        snapshot_download("dwzhu/e5-base-4k",
                          repo_type="model",
                          cache_dir=str(cache_dir),
                          local_dir=str(target_dir))

    def load_model(self):
        model_path = str(self.base_path / "e5-base-4k")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa")
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def load_dataset(self):
        # Check existing shards, skip ahead for processing
        if len(list(self.embd_dir.glob("shard_*.parquet"))) > 52:
            skip_items_count = (52 * (self.config.tokenization_batch_size * 8192)) # First 52 shards are large
            skip_items_count += ((len(list(self.embd_dir.glob("shard_*.parquet")))-52) * self.config.shard_size) # Everything beyond shard 52 is smaller
        else:
            # If we test in some other dir, we just run with the standard shard size
            skip_items_count = (len(list(self.embd_dir.glob("shard_*.parquet"))) * self.config.shard_size)
        dataset = load_dataset("parquet",
                               data_files={"train": str(self.base_path / "Pile_Deduplicated" / "data" / "train-*.parquet")},
                               cache_dir=None,
                               split="train",
                               streaming=True)
        if self.accelerator.is_main_process:
            print(f"Skipping {skip_items_count} items. This may take a while...")
        dataset = dataset.skip(skip_items_count)
        return dataset 
    
    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = attention_mask.to(last_hidden_states.device)
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @torch.no_grad()
    def process_batch(self, tokenized_batch):
        inputs = {k: v.to(self.accelerator.device) for k, v in tokenized_batch.items()}
        outputs = self.model(**inputs)
        embeddings = self.average_pool(outputs.last_hidden_state, tokenized_batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy()
        del embeddings, inputs, outputs, tokenized_batch
        return embeddings_np
    
    def run(self):
        self.download_model()
        self.load_model()
        dataset = self.load_dataset()
        
        writer = AsyncWriter(self.embd_dir, self.config)

        # Create queues for tokenization and inference
        tokenization_input_queue = queue.Queue(maxsize=self.config.prefetch_batches)
        tokenization_output_queue = queue.Queue(maxsize=self.config.prefetch_batches)

        # Create and start tokenization workers
        tokenization_workers = []
        for _ in range(self.config.num_worker_threads):
            worker = TokenizationWorker(self.tokenizer, self.config, tokenization_input_queue, tokenization_output_queue)
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            tokenization_workers.append(thread)

        # Function to feed data to tokenization workers
        def feed_tokenization_workers():
            for batch in dataset.iter(batch_size=self.config.tokenization_batch_size):
                tokenization_input_queue.put(batch)
            for _ in range(self.config.num_worker_threads):
                tokenization_input_queue.put(None)

        # Start feeding data to tokenization workers
        feed_thread = threading.Thread(target=feed_tokenization_workers, daemon=True)
        feed_thread.start()

        # Process batches
        current_shard_embeddings = []
        current_shard_texts = []
        
        # We don't know the total number of batches in advance
        pbar = tqdm(total=None)
        
        while True:
            tokenized_batch = tokenization_output_queue.get()
            if tokenized_batch is None:
                break

            tokenized, original_texts = tokenized_batch
            embeddings = self.process_batch(tokenized)
            
            current_shard_embeddings.extend(embeddings)
            current_shard_texts.extend(original_texts)

            # Write shard when it reaches the target size
            if len(current_shard_embeddings) >= self.config.shard_size:
                if self.accelerator.is_main_process:
                    writer.write(current_shard_embeddings[:], current_shard_texts[:])
                # Wait for the writer to finish this shard
                writer.write_queue.join()
                
                # Now it's safe to delete/clear the lists (as the using writer finished)
                del current_shard_embeddings[:]
                del current_shard_texts[:]
                torch.cuda.empty_cache()
                gc.collect()

            pbar.update(len(embeddings))
            del tokenized_batch, tokenized, embeddings, original_texts

        # Write remaining data
        if current_shard_embeddings and self.accelerator.is_main_process:
            writer.write(current_shard_embeddings, current_shard_texts)
        
        # Cleanup
        writer.finish()
        self.accelerator.wait_for_everyone()

        # Wait for all tokenization workers to finish
        for worker in tokenization_workers:
            worker.join()

        # Signal to cluster process
        if self.accelerator.is_main_process:
            end_file = self.embd_dir / "End_Here.txt"
            end_file.touch()

        pbar.close()

if __name__ == "__main__":
    config = Config()
    pipeline = EmbeddingPipeline(config)
    pipeline.run()

# tmux new -s embed_pile
# conda activate minicorpus
# accelerate launch --mixed_precision fp16 --num_processes=1 03_embed_pile_dedup_turbo.py
# You can run this as multi-card process, but ... that won't do anything. Only one process persists.
# Detach from tmux session: Ctrl-B followed by D
# Reattach to tmux session: tmux attach -t embed_pile
# tmux list-sessions
# tmux kill-session -t embed_pile
#
# This took roughly 170 hours.