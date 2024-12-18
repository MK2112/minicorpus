import os
import torch
import queue
import threading
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# Do not use this script. I left it here purely for reference.
# This is what naivite looks like. The script is slow and inefficient and crashes with OOM.
# Use the turbo version instead: 03_embed_pile_dedup_turbo.py

@dataclass
class Config:
    base_dir: str = "/vol/tmp/koppelmm"
    batch_size: int = 8192
    prefetch_batches: int = 8
    embedding_dim: int = 1024
    shard_size: int = batch_size * 256 # Embeddings per shard
    num_worker_threads: int = 8
    
class AsyncWriter:
    def __init__(self, output_dir: Path, config: Config):
        self.output_dir = output_dir
        self.config = config
        self.write_queue = queue.Queue(maxsize=4)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        self.current_shard = 0
        
    def _writer_loop(self):
        while True:
            try:
                data = self.write_queue.get()
                if data is None:
                    break
                    
                embeddings, texts = data
                table = pa.Table.from_arrays(
                    [pa.array(embeddings), pa.array(texts)],
                    names=['embedding', 'text']
                )
                
                shard_path = self.output_dir / f"shard_{self.current_shard:09d}.parquet"
                pq.write_table(table, str(shard_path))
                self.current_shard += 1
                
            finally:
                self.write_queue.task_done()
                
    def write(self, embeddings: List[np.ndarray], texts: List[str]):
        self.write_queue.put((embeddings, texts))
        
    def finish(self):
        self.write_queue.put(None)
        self.writer_thread.join()

class EmbeddingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.base_path = Path(config.base_dir)
        self.accelerator = Accelerator()
        self.setup_directories()
        
    def setup_directories(self):
        self.embd_dir = self.base_path / "Pile_Deduplicated_Embedded"
        self.embd_dir.mkdir(exist_ok=True)
        
    def download_model(self):
        target_dir = self.base_path / "e5-large-v2"
        cache_dir = self.base_path / "e5-large-v2_Cache"
        
        for dir_path in [target_dir, cache_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        snapshot_download(
            "intfloat/e5-large-v2",
            repo_type="model",
            cache_dir=str(cache_dir),
            local_dir=str(target_dir)
        )
        
    def load_model(self):
        model_path = str(self.base_path / "e5-large-v2")
        self.model = SentenceTransformer(model_path, local_files_only=True)
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        
    def load_dataset(self):
        return load_dataset(
            "parquet",
            data_files={"train": str(self.base_path / "Pile_Deduplicated" / "data" / "train-*.parquet")},
            cache_dir=str(self.base_path / "Pile_Deduplicated_Cache"),
            split="train",
            streaming=True
        )
    
    @torch.no_grad()
    def process_batch(self, batch: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        # Add "query: " prefix to each text in the batch
        # Not doing this is claimed to bring performance/distinctiveness degradation
        prefixed_texts = ["query: " + text for text in batch['text']]
        embeddings = self.model.encode(
            prefixed_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True 
        )
        return embeddings
    
    def run(self):
        self.download_model()
        self.load_model()
        dataset = self.load_dataset()
        
        # Set up async writer
        writer = AsyncWriter(self.embd_dir, self.config)
        
        # Create data loader with prefetching
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_worker_threads,
            prefetch_factor=self.config.prefetch_batches,
            pin_memory=True
        )
        
        # Process batches
        current_shard_embeddings = []
        current_shard_texts = []
        
        for batch in tqdm(dataloader):
            embeddings = self.process_batch(batch)
            
            current_shard_embeddings.extend(embeddings)
            current_shard_texts.extend(batch['text']) # no prefix here
            
            # Write shard when it reaches the target size
            if len(current_shard_embeddings) >= self.config.shard_size:
                if self.accelerator.is_main_process:
                    writer.write(current_shard_embeddings, current_shard_texts)
                current_shard_embeddings = []
                current_shard_texts = []
        
        # Write remaining data
        if current_shard_embeddings and self.accelerator.is_main_process:
            writer.write(current_shard_embeddings, current_shard_texts)
        
        # Cleanup
        writer.finish()
        self.accelerator.wait_for_everyone()

if __name__ == "__main__":
    config = Config()
    pipeline = EmbeddingPipeline(config)
    pipeline.run()

# tmux new -s embed_pile
# conda activate minipile
# accelerate launch --multi_gpu --gpu_ids 2,3 --mixed_precision fp16 --num_processes=2 03_embed_pile_dedup.py
# DONT DO: accelerate launch --multi_gpu --mixed_precision fp16 --num_processes=4 03_embed_pile_dedup.py # maxes out at 38% usage per card
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t embed_pile
# tmux list-sessions
# tmux kill-session -t embed_pile