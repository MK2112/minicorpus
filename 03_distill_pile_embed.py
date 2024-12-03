import gc
import json
import numpy as np
import jsonlines
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, Dict, List
from multiprocessing import Pool, Manager, cpu_count

@dataclass
class DistillConfig:
    base_dir: Path = Path("/vol/tmp/koppelmm")
    cluster_dir: Path = base_dir / "MiniPile_BatchKMeans/clustering_sorted"
    cluster_info_path: Path = base_dir / "MiniPile_BatchKMeans/clustering_results/cluster_info_for_inspection.json"
    embd_dir: Path = base_dir / "Pile_Deduplicated_Embd"
    num_clusters: int = 220 # As per paper
    num_clusters_to_exclude: int = 38 # As per paper
    edition: str = "Self" # Version of MiniPile, distinguishes file naming + output directory
    excluded_clusters: Set[int] = field(default_factory=lambda: {10, 15, 16, 22, 26, 28, 35, 37, 39, 40, 44, 46, 
                                   51, 57, 61, 64, 78, 86, 87, 88, 90, 94, 99, 101,
                                   102, 103, 111, 114, 152, 155, 163, 166, 167, 181,
                                   196, 200, 218, 219})
    train_count: int = 1_000_000
    val_count: int = 500
    test_count: int = 10_000 # 1M/500/10k Train/Val/Test total
    examples_per_cluster: int = int((train_count + val_count + test_count) / (num_clusters - num_clusters_to_exclude))
    output_shard_size: int = 100_000
    rng: np.random.Generator = np.random.default_rng(42) # reproducibility
    
    def __post_init__(self):
        # Just nicer and more distinct to place here
        self.output_dir = self.base_dir / f"MiniPile_{self.edition}"
        self.output_dir.mkdir(exist_ok=True, parents=True)

class MiniPileDistiller:
    def __init__(self, config: DistillConfig):
        self.config = config
        # Validate configuration parameters for cluster exclusion
        if len(self.config.excluded_clusters) != self.config.num_clusters_to_exclude:
            raise ValueError(f"Must exclude exactly {self.config.num_clusters_to_exclude} clusters to adhere to the paper. You provided {len(self.config.excluded_clusters)} clusters.")
        self._load_total_cluster_info()
        self._compute_shard_scopes()
    
    def _load_total_cluster_info(self):
        # Load general cluster information JSON file, populate class attribute
        with open(self.config.cluster_info_path, 'r') as f:
            self.cluster_info = json.load(f)
    
    def _compute_shard_scopes(self):
        # Precompute cumulative entries for efficient document lookup
        self.shard_idxs = []
        cumulative_idxs = 0
        num_shards = len(list(Path(self.config.embd_dir).glob("shard_*.parquet")))
        
        # First 52 shards have 1,048,576 entries
        # Subsequent shards have 524,288 entries
        # Last shard has (134,318,121 - 52 * 1,048,576 - 152 * 524,288) = 100,393 entries
        for _ in range(52):
            self.shard_idxs.append(cumulative_idxs)
            cumulative_idxs += 1_048_576
        
        for _ in range(52, num_shards - 1):
            self.shard_idxs.append(cumulative_idxs)
            cumulative_idxs += 524_288

        last_shard_size = 134_318_121 - cumulative_idxs
        if last_shard_size > 0:
            self.shard_idxs.append(cumulative_idxs)

    def _get_idxs_for_cluster(self, cluster_idx: int) -> List[int]:
        # Get document indices assoricated with a given cluster
        cluster_file = self.config.cluster_dir / f"cluster_{cluster_idx:03d}.jsonl"
        cluster_indices = []
        with jsonlines.open(cluster_file) as reader:
            for entry in reader:
                if entry['cluster'] == cluster_idx:
                    cluster_indices.append(entry['idx'])
        return cluster_indices
    
    def _sample_cluster_docs(self, valid_indices: List[int]) -> List[int]:
        # Sample documents from cluster, *explicitly* randomly
        # valid_indices are already somewhat shuffled, but we really make sure here, I'm paranoid, that's also why I made it its own function
        # The paper does not explicitly state whether cluster-size-proportioned sampling was applied
        #
        # I see it like this:
        # While MiniPile doesn't prohibit proportional sampling, it does not mention it as part of the methodology either.
        # I interpret that any adjustments made for that can be seen as extending or refining the methodology rather than replicating it as stated.
        # The paper does not mention proportional sampling or any attempt to adjust the example count sampled based on cluster sizes.
        # This omission suggests a primary focus was simplicity and achieving a set, fixed (1M/500/10k) dataset size, rather than perfect proportional representation.
        # Proportional sampling would moreover make it more difficult to hit the specific dataset size goal with the parts of the pipeline that are in fact described.
        #
        # Still, I just have to acknowledge that proportional sampling could have clear advantages here.
        return self.config.rng.choice(valid_indices, size=min(len(valid_indices), self.config.examples_per_cluster), replace=False).tolist()
    
    def _shard_with_idx(self, idx: int) -> int:
        # Find the shard containing a specific index by entry count heuristic
        for i, _ in enumerate(self.shard_idxs):
            if i + 1 < len(self.shard_idxs) and idx < self.shard_idxs[i + 1]:
                return i
        return len(self.shard_idxs) - 1

    def _shuffle_split(self, total_docs: int) -> Dict[str, List[int]]:
        # Chop up for train/val/test splits
        all_indices = list(range(total_docs))
        # I want at all cost to avoid having consecutive indices from the same cluster overly represented in the same split
        self.config.rng.shuffle(all_indices)
        return {'train': all_indices[:self.config.train_count],
                'validation': all_indices[self.config.train_count:(self.config.train_count + self.config.val_count)],
                'test': all_indices[(self.config.train_count + self.config.val_count):(self.config.train_count + self.config.val_count + self.config.test_count)]}
    
    def build_minipile(self):
        # Prepare list of document indices to extract
        total_doc_indices_to_extract = set()
        for cluster in tqdm(range(self.config.num_clusters), desc="Assembling Indices to Extract", unit="cluster"):
            if cluster not in self.config.excluded_clusters:
                total_doc_indices_to_extract.update(self._sample_cluster_docs(self._get_idxs_for_cluster(cluster)))
        
        # Generate data splits
        print(f"[+] Total documents to extract: {len(total_doc_indices_to_extract)}. Shuffling and splitting.")
        data_splits = self._shuffle_split(len(total_doc_indices_to_extract))
        
        # Process and write documents in splits, in parallel, in the sun, with a drink
        for split_name, split_indices in tqdm(data_splits.items(), desc="Processing Splits", unit="split"):
            self._process_split(total_doc_indices_to_extract, split_indices, split_name)
        
        print(f"[+] MiniPile created with {len(total_doc_indices_to_extract)} documents across train/val/test splits.")
    
    def _process_split(self, all_indices: Set[int], split_indices: List[int], split_name: str):
        # Group indices by shard
        indices_by_shard = {}
        for idx in tqdm(split_indices, desc=f"Split {split_name} Shard Grouping", unit="doc"):
            if idx in all_indices:
                shard_idx = self._shard_with_idx(idx)
                indices_by_shard.setdefault(shard_idx, []).append(idx)

        parquet_files = sorted(Path(self.config.embd_dir).glob("shard_*.parquet")) # 9-digit zero-padded shard indices in filename

        with Manager() as manager:
            shard_counter = manager.Value('i', 0)  # Shared counter
            counter_lock = manager.Lock()  # Lock to ensure safe increments
            tasks = [(shard_idx, indices, parquet_files[shard_idx], split_name, self.config, self.shard_idxs, shard_counter, counter_lock) for shard_idx, indices in indices_by_shard.items()]
            # Process shards #Parallelly
            with Pool(cpu_count() // 2) as pool:
                pool.starmap(self._process_shard, tasks)

    @staticmethod
    def _process_shard(shard_idx: int, indices_to_extract: List[int], file_path: Path, split_name: str, config: DistillConfig,
                       shard_idxs: List[int], shard_counter, counter_lock):
        # Convert indices_to_extract to local shard indices
        local_indices = [idx - shard_idxs[shard_idx] for idx in indices_to_extract]
        
        # Create a pyarrow dataset and apply filtering
        parquet_ds = ds.dataset(file_path)
        filter_expr = pa.compute.is_in(
            pa.array(local_indices), parquet_ds.schema.field("idx")
        )
        table = parquet_ds.to_table(filter=filter_expr, columns=["idx", "text"])
        
        # Process filtered rows directly
        current_shard_docs = [{"idx": row["idx"].as_py(), "text": row["text"].as_py()} for row in table]
        
        # Write out shards based on the configured shard size
        num_docs = len(current_shard_docs)
        with tqdm(total=num_docs, desc=f"Processing Shard {shard_idx}", unit="doc") as pbar:
            # Write out shards based on the configured shard size
            while len(current_shard_docs) >= config.output_shard_size:
                with counter_lock:
                    shard_id = shard_counter.value
                    shard_counter.value += 1
                
                MiniPileDistiller._write_parquet_shard(current_shard_docs[:config.output_shard_size], shard_id, split_name, config)
                del current_shard_docs[:config.output_shard_size]
                gc.collect()
                pbar.update(config.output_shard_size)

            # Handle remaining documents
            if current_shard_docs:
                with counter_lock:
                    shard_id = shard_counter.value
                    shard_counter.value += 1
                
                MiniPileDistiller._write_parquet_shard(current_shard_docs, shard_id, split_name, config)
                del current_shard_docs[:]
                gc.collect()
                pbar.update(len(current_shard_docs))

        # Handle remaining documents
        if current_shard_docs:
            with counter_lock:
                shard_id = shard_counter.value
                shard_counter.value += 1
            
            MiniPileDistiller._write_parquet_shard(current_shard_docs, shard_id, split_name, config)
            del current_shard_docs
            gc.collect()

    @staticmethod
    def _write_parquet_shard(docs: List[Dict], shard_index: int, split_name: str, config: DistillConfig):
        # Write a shard of documents to a Parquet file
        df = pa.table({'text': [doc['text'] for doc in docs], 'pile_idx': [doc['idx'] for doc in docs]})
        output_path = config.output_dir / f"minipile_{config.edition}_{split_name}_shard_{shard_index:09d}.parquet"
        pq.write_table(df, output_path)
        del df

if __name__ == "__main__":
    config = DistillConfig()
    print("[!] Loaded MiniPile Creation Config")
    distiller = MiniPileDistiller(config)
    print("[!] Initialized Distiller")
    distiller.build_minipile()

# tmux new -s minipile
# conda activate minipile
# (pip install jsonlines)
# python 03_distill_pile_embed.py
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t minipile
# tmux list-sessions
# tmux kill-session -t minipile