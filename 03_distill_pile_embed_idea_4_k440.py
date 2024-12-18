import gc
import json
import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from typing import Set, Dict, List
from fastparquet import ParquetFile
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count

@dataclass
class DistillConfig:
    base_dir: Path = Path("/vol/tmp/koppelmm")
    cluster_dir: Path = base_dir / "MiniPile_BatchKMeans_Double/clustering_sorted"
    cluster_info_path: Path = base_dir / "MiniPile_BatchKMeans_Double/clustering_results/cluster_info_for_inspection.json"
    embd_dir: Path = base_dir / "Pile_Deduplicated_Embd"
    num_clusters: int = 440 # As per paper
    num_clusters_to_exclude: int = 70 # As per paper
    edition: str = "k440_fixed" # Version of MiniPile, distinguishes file naming + output directory
    excluded_clusters: Set[int] = field(default_factory=lambda: {3, 6, 7, 8, 9, 19, 20, 24, 26, 32, 39, 46, 47, 48, 49, 50, 51, 54, 57, 60, 69,
                                                                 85, 89, 91, 92, 106, 108, 111, 120, 144, 147, 152, 162, 165, 169, 172, 176, 178,
                                                                 189, 199, 203, 208, 210, 212, 215, 216, 224, 227, 243, 244, 254, 264, 274, 283,
                                                                 289, 299, 304, 322, 333, 338, 350, 369, 370, 388, 401, 404, 410, 413, 417, 430})
    train_count: int = 1_000_000
    val_count: int = 500
    test_count: int = 12_663 # 1M/500/10k Train/Val/Test total (I don't now why exactly, but nudging test +2,663 makes it 1M/500/10k in the end)
    examples_per_cluster: int = (train_count + val_count + test_count) // (num_clusters - num_clusters_to_exclude)
    extra_examples: int = (train_count + val_count + test_count) % (num_clusters - num_clusters_to_exclude)
    output_shard_size: int = 100_000
    pile_size: int = 134_318_121 # Could've been read from the metadata file, too
    rng: np.random.Generator = np.random.default_rng(42) # reproducibility

    def __post_init__(self):
        # Just nicer and more distinct to place here
        self.output_dir = self.base_dir / f"MiniPile_{self.edition}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        # Not used here, but needed later to not hit disk quotas
        self.cache_dir = self.base_dir / f"MiniPile_{self.edition}_Cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        # Distribute the needed extra examples to be drawn from across the clusters most evenly
        self.cluster_extra_samples = [0] * self.num_clusters
        extra_examples_left = self.extra_examples
        # Distribute extra examples across non-excluded clusters
        for cluster in range(self.num_clusters):
            if cluster not in self.excluded_clusters and extra_examples_left > 0:
                self.cluster_extra_samples[cluster] = 1
                extra_examples_left -= 1
        # Verify all extra examples are distributed
        assert extra_examples_left == 0, f"Failed to distribute all extra examples: {extra_examples_left} left"

class MiniCorpusWriter:
    def __init__(self, output_dir: Path, edition: str, output_shard_size: int):
        self.output_dir = output_dir
        self.edition = edition
        self.output_shard_size = output_shard_size
        # Buffers for each split
        self.buffers = { split : [] for split in ['train', 'validation', 'test'] }
        # Shard counters for each split
        self.shard_counters = { split : 0 for split in ['train', 'validation', 'test'] }
    
    def _write_shard(self, split: str, force: bool = False):
        buffer = self.buffers[split]
        print(f"[~] Writing {split} shard {self.shard_counters[split]}")
        if force or len(buffer) >= self.output_shard_size:
            # Determine how many documents to write
            docs_to_write = buffer[:self.output_shard_size] if not force else buffer
            df = pa.table({'text': [doc['text'] for doc in docs_to_write], 'pile_idx': [doc['idx'] for doc in docs_to_write]})
            # Write Parquet file
            output_path = self.output_dir / f"minipile_{self.edition}_{split}_shard_{self.shard_counters[split]:09d}.parquet"
            pq.write_table(df, output_path)
            # Update buffer and counter
            self.buffers[split] = buffer[len(docs_to_write):]
            self.shard_counters[split] += 1
            del df
            gc.collect()
    
    def add_document(self, doc: Dict, split: str):
        self.buffers[split].append(doc)
        # Write shard if buffer reaches output_shard_size
        if len(self.buffers[split]) >= self.output_shard_size:
            self._write_shard(split)
    
    def finalize(self):
        for split in self.buffers:
            if self.buffers[split]:
                self._write_shard(split, force=True)

class MiniCorpusDistiller:
    def __init__(self, config: DistillConfig):
        self.config = config
        # Validate configuration parameters for cluster exclusion
        if len(self.config.excluded_clusters) != self.config.num_clusters_to_exclude:
            raise ValueError(f"Must exclude {self.config.num_clusters_to_exclude} clusters. You provided {len(self.config.excluded_clusters)} clusters.")
        self._load_total_cluster_info()
        self._compute_shard_scopes()
        self.shard_counter: int = 0
        self.writer = MiniCorpusWriter(output_dir=config.output_dir,
                                       edition=config.edition,
                                       output_shard_size=config.output_shard_size)
    
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

        last_shard_size = self.config.pile_size - cumulative_idxs
        if last_shard_size > 0:
            self.shard_idxs.append(cumulative_idxs)


    def _read_idxs_for_cluster(self, cluster_idx: int) -> List[int]:
        # Read document indices assoricated with a given cluster
        cluster_file = self.config.cluster_dir / f"cluster_{cluster_idx:03d}.jsonl"
        cluster_indices = []
        with jsonlines.open(cluster_file) as reader:
            for entry in reader:
                if entry['cluster'] == cluster_idx:
                    cluster_indices.append(entry['idx'])
        return cluster_indices
    
    def _sample_cluster_docs(self, valid_indices: List[int], num_samples: int) -> List[int]:
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
        return self.config.rng.choice(valid_indices, size=min(len(valid_indices), num_samples), replace=False).tolist()
    
    def _shard_with_idx(self, idx: int) -> int:
        # Find the shard containing a specific index by entry count heuristic
        # Return the shard index as well as the local index within the shard
        for i, _ in enumerate(self.shard_idxs):
            if i + 1 < len(self.shard_idxs) and idx < self.shard_idxs[i + 1]:
                return i
        return len(self.shard_idxs) - 1

    def _shuffle_split(self, idxs_to_shuffle: List[int]) -> Dict[str, List[int]]:
        # Shuffle all_indices, split for train/val/test splits
        self.config.rng.shuffle(idxs_to_shuffle)
        return {'train': idxs_to_shuffle[:self.config.train_count],
                'validation': idxs_to_shuffle[self.config.train_count:(self.config.train_count + self.config.val_count)],
                'test': idxs_to_shuffle[(self.config.train_count + self.config.val_count):(self.config.train_count + self.config.val_count + self.config.test_count)]}
    
    def _process_cluster(self, cluster: int) -> Set[int]:
        if cluster not in self.config.excluded_clusters:
            # Get document indices associcated with cluster
            valid_indices = self._read_idxs_for_cluster(cluster)
            # We want to extract this many examples from the cluster by random sampling
            return set(self._sample_cluster_docs(valid_indices, self.config.examples_per_cluster + self.config.cluster_extra_samples[cluster]))
        else:
            return set()

    def build_minicorpus(self):
        with Pool(processes=cpu_count() // 2) as pool:
            results = pool.map(self._process_cluster, range(self.config.num_clusters))
        # Combine results from all workers
        idxs_to_extract = sorted(set().union(*results))
        len_idxs_to_extract = len(idxs_to_extract)
        # This should show a nice distance between the first and last document index, roughly spanning Pile dataset size
        print(f"[Info] Min document idx: {idxs_to_extract[0]}, Max document idx: {idxs_to_extract[-1]}")
        print(f"[+] Total documents to extract: {len_idxs_to_extract}. Shuffling and splitting.")
        # I want at all cost to avoid having consecutive indices from the same cluster overly represented in the same split
        data_splits = self._shuffle_split(idxs_to_extract)
        del idxs_to_extract
        # Process and write documents in splits
        for split_name, split_idxs_to_extract in tqdm(data_splits.items(), desc="Processing Splits", unit="split"):
            self._process_split(split_idxs_to_extract, split_name)
        print(f"[+] MiniPile created with {len_idxs_to_extract} documents across train/val/test splits.")
    
    def _process_split(self, split_idxs_to_extract: List[int], split_name: str):
        # Convert split_indices to a set for faster lookup
        split_idx_set = set(split_idxs_to_extract)
        # Group indices by shard
        idxs_by_shard = {}
        # Note which Pile shard each document is in
        for idx in tqdm(split_idxs_to_extract, desc=f"Split {split_name} Shard Grouping", unit="doc"):
            shard_idx = self._shard_with_idx(idx)
            if shard_idx not in idxs_by_shard:
                idxs_by_shard[shard_idx] = []
            if idx in split_idx_set:
                idxs_by_shard[shard_idx].append(idx)

        # The sum of all indices in idxs_by_shard should equal the total number of indices to extract
        assert sum(len(indices) for indices in idxs_by_shard.values()) == len(split_idxs_to_extract)
        
        # These are the Parquet files that contain the documents
        parquet_files = sorted(Path(self.config.embd_dir).glob("shard_*.parquet"))
        for shard_idx, indices in tqdm(idxs_by_shard.items(), desc=f"Processing {split_name} Shards", unit="shard"):
            # Read the indices accumulated for each shard and persist them to a new Parquet file
            self._process_shard(shard_idx, indices, parquet_files[shard_idx], split_name)
        
        del idxs_by_shard
        # Pack up your stuff, we're done with the split
        self.writer.finalize()

    def _read_fast_parquet(self, file_path: str, idxs: List[int]) -> np.ndarray:
        # This is really fast, really memory-optimized, but craps all over the cache; I can't control that.
        parquet = ParquetFile(file_path) # https://github.com/dask/fastparquet/issues/386
        result = parquet.to_pandas(columns=['text'])['text'].to_numpy()[idxs]
        del parquet
        return result

    def _process_shard(self, shard_idx: int, idxs_to_extract: List[int], file_path: Path, split_name: str) -> None:
        # Determine shard-local offsets from the global indices
        idxs_to_extract = sorted(idxs_to_extract)
        local_idxs = [idx - self.shard_idxs[shard_idx] for idx in idxs_to_extract]
        
        assert all(idx >= 0 for idx in local_idxs), f"Negative local index in shard {shard_idx}: {local_idxs}"
        assert len(local_idxs) == len(set(local_idxs)), f"Duplicate local indices in shard {shard_idx}"
        
        # Read the entire Parquet file
        # (https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html is actually usable)
        # I needed a compromise for speed and memory usage, so I read the entire file as efficiently as possible, but immediately
        # selected the columns and filter the entries I actually need.
        # This might be the funniest change I made throughout the project, but pandas slowed things down by ~10%.
        selected_texts = self._read_fast_parquet(str(file_path), local_idxs)
        
        # Merging the global indices back in for reference/debugging/fun
        current_shard_docs = [{'text': text, 'idx': idx} for text, idx in zip(selected_texts, idxs_to_extract)]
        del selected_texts, local_idxs, idxs_to_extract
        
        # Write out shards based on the configured shard size
        # This is butchered but works, still
        with tqdm(total=len(current_shard_docs), desc=f"Processing Shard {shard_idx}", unit="doc") as pbar:
            for doc in current_shard_docs:
                self.writer.add_document(doc, split_name)
                pbar.update(1)
        del current_shard_docs
        gc.collect()

if __name__ == "__main__":
    config = DistillConfig()
    distiller = MiniCorpusDistiller(config)
    distiller.build_minicorpus()

# tmux new -s mini_k440
# conda activate minipile
# (pip install jsonlines)
# python 03_distill_pile_embed_idea_4_k440.py
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t mini_k440
# tmux list-sessions
# tmux kill-session -t mini_k440
#
# 