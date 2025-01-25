import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
import queue
import threading
import pyarrow as pa
import pyarrow.parquet as pq
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import filelock

@dataclass
class Config:
    base_dir: str = "/vol/tmp/koppelmm"
    tokenization_batch_size: int = 128
    prefetch_batches: int = 2
    shard_size: int = tokenization_batch_size * 8192  # ~1M embeddings per shard
    num_worker_threads: int = 4
    max_length: int = 1024   # Each embedding is 768-dimensional
    # Set these for sectioning what you want to process
    start_shard: int = None  # Starting shard index (inclusive), trying out this Optional thingy for once
    end_shard: int = None    # Ending shard index (exclusive)

class AsyncWriter:
    def __init__(self, output_dir: Path, config: Config):
        self.output_dir = output_dir
        self.config = config
        self.write_queue = queue.Queue(maxsize=4)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        # Start counting from the specified start_shard if provided
        self.current_shard = (self.config.start_shard if self.config.start_shard is not None else self._get_next_shard_index())

    def _get_next_shard_index(self):
        # Count existing shards, determine the follow-up shard index
        existing_shards = list(self.output_dir.glob("shard_*.parquet"))
        return len(existing_shards)
        
    def _writer_loop(self):
        while True:
            try:
                data = self.write_queue.get()
                if data is None:
                    break
                    
                embeddings, texts = data
                table = pa.Table.from_arrays([pa.array(embeddings), pa.array(texts)], names=['embedding', 'content'])
                
                # Use file lock to prevent concurrent writes to the same directory
                # We're prob going overkill with this, but this is the textbook way to do it and, eh, trying it out i guess
                lock_path = self.output_dir / "write.lock"
                with filelock.FileLock(str(lock_path)):
                    shard_path = self.output_dir / f"shard_{self.current_shard:09d}.parquet"
                    pq.write_table(table, str(shard_path))
                    self.current_shard += 1

                del embeddings, texts, table
            finally:
                self.write_queue.task_done()
                
    def write(self, embeddings: List[np.ndarray], texts: List[str]):
        # Only write if we're within our assigned shard range
        if (self.config.end_shard is None or self.current_shard < self.config.end_shard):
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
            
            # Come to think of it, maybe this prefix adding akin to what the original model description
            # may be another explanation for why we perform well even with the small E5-Base-4k
            prefixed_texts = ["query: " + text for text in batch['content']]
            tokenized = self.tokenizer(prefixed_texts, 
                                       max_length=self.config.max_length, 
                                       padding="max_length",
                                       truncation=True,
                                       return_tensors='pt')
            self.output_queue.put((tokenized, batch['content']))
            del prefixed_texts

class EmbeddingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.base_path = Path(config.base_dir)
        self.accelerator = Accelerator()
        self.setup_directories()
        
    def setup_directories(self):
        self.embd_dir = self.base_path / "RefinedWeb_Embd"
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
        # Calculate skip based on start_shard if specified
        skip_items_count = (self.config.start_shard * self.config.shard_size if self.config.start_shard is not None else 0)
        dataset = load_dataset("parquet",
                               data_files={"train": str(self.base_path / "RefinedWeb" / "data" / "train-*.parquet")},
                               cache_dir=None,
                               split="train",
                               streaming=True)
        if skip_items_count > 0 and self.accelerator.is_main_process:
            print(f"Skipping {skip_items_count} items to start from shard {self.config.start_shard}")
        dataset = dataset.skip(skip_items_count)
        # If end_shard specified, calculate how many items to take
        if self.config.end_shard is not None:
            items_to_take = ((self.config.end_shard - (self.config.start_shard or 0)) * self.config.shard_size)
            dataset = dataset.take(items_to_take)
        return dataset
    
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
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

        tokenization_input_queue = queue.Queue(maxsize=self.config.prefetch_batches)
        tokenization_output_queue = queue.Queue(maxsize=self.config.prefetch_batches)

        tokenization_workers = []
        for _ in range(self.config.num_worker_threads):
            worker = TokenizationWorker(self.tokenizer, 
                                        self.config,
                                        tokenization_input_queue,
                                        tokenization_output_queue)
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            tokenization_workers.append(thread)

        def feed_tokenization_workers():
            for batch in dataset.iter(batch_size=self.config.tokenization_batch_size):
                tokenization_input_queue.put(batch)
            for _ in range(self.config.num_worker_threads):
                tokenization_input_queue.put(None)

        feed_thread = threading.Thread(target=feed_tokenization_workers, daemon=True)
        feed_thread.start()

        current_shard_embeddings = []
        current_shard_texts = []
        
        pbar = tqdm(total=None)
        
        while True:
            tokenized_batch = tokenization_output_queue.get()
            if tokenized_batch is None:
                break

            tokenized, original_texts = tokenized_batch
            embeddings = self.process_batch(tokenized)
            
            current_shard_embeddings.extend(embeddings)
            current_shard_texts.extend(original_texts)

            if len(current_shard_embeddings) >= self.config.shard_size:
                if self.accelerator.is_main_process:
                    writer.write(current_shard_embeddings[:], current_shard_texts[:])
                writer.write_queue.join()
                del current_shard_embeddings[:]
                del current_shard_texts[:]
                torch.cuda.empty_cache()
                gc.collect()

            pbar.update(len(embeddings))
            del tokenized_batch, tokenized, embeddings, original_texts

        if current_shard_embeddings and self.accelerator.is_main_process:
            writer.write(current_shard_embeddings, current_shard_texts)
        
        writer.finish()
        self.accelerator.wait_for_everyone()

        for worker in tokenization_workers:
            worker.join()

        if self.accelerator.is_main_process:
            # Create a range-aware completion notice (file)
            # Maybe I need that, maybe not, better safe than sorry
            range_str = f"{self.config.start_shard}-{self.config.end_shard}"
            end_file = self.embd_dir / f"completed_range_{range_str}.txt"
            end_file.touch()

        pbar.close()

if __name__ == "__main__":    
    EmbeddingPipeline(Config()).run()