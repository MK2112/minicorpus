import json
import numpy as np
from pathlib import Path
from typing import Set, Dict, List
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import defaultdict
import random

@dataclass
class FilterConfig:
    base_dir: Path = Path("/vol/tmp/koppelmm")
    batch_size: int = 16384                                 # Batch size for processing embeddings
    num_clusters: int = 220                                 # Total number of clusters
    num_clusters_to_exclude: int = 38                       # Clusters to exclude
    examples_per_cluster: int = int(1010500 / (num_clusters - num_clusters_to_exclude)) # Target examples per remaining cluster
    output_shard_size: int = 100_000                        # Examples per output shard
    
    def __post_init__(self):
        self.embd_dir = self.base_dir / "Pile_Deduplicated_Embd"    # Dataset w/ embeddings
        self.cluster_dir = self.base_dir / "MiniPile_BatchKMeans"   # Clustering results
        self.output_dir = self.base_dir / "MiniPile_Self"           # Target output MiniPile dataset
        self.output_dir.mkdir(exist_ok=True, parents=True)

class MiniPileFilter:
    def __init__(self, config: FilterConfig):
        self.config = config
        self.excluded_clusters: Set[int] = set() # Set of excluded cluster IDs
        self._load_cluster_info()
    
    def _load_cluster_info(self) -> None:
        with open(self.config.cluster_dir / "cluster_info_for_inspection.json", "r") as f:
            self.cluster_info = json.load(f)
    
    def exclude_clusters(self, cluster_ids: List[int]) -> None:
        self.excluded_clusters = set(cluster_ids)
        if len(self.excluded_clusters) != self.config.num_clusters_to_exclude:
            print(f"[!] Must exclude exactly {self.config.num_clusters_to_exclude} clusters to adhere to the paper. You provided {len(self.excluded_clusters)} clusters.")
        with open(self.config.output_dir / "excluded_clusters.json", "w") as f:
            json.dump({"excluded_clusters": sorted(list(self.excluded_clusters))}, f, indent=2)
    
    def _get_cluster_sample_indices(self) -> Dict[int, Set[int]]:
        results_dir = self.config.cluster_dir / "clustering_results"
        cluster_indices = defaultdict(list)  # Map cluster_id -> list of document indices (will be ~3.78 GB in memory)
        sampled_indices = {}
        # Aggregate indices for each cluster
        for chunk_file in tqdm(sorted(results_dir.glob("cluster_results_chunk_*.jsonl")), desc="Aggregating cluster indices"):
            with open(chunk_file, 'r') as f:
                for idx, line in enumerate(f):
                    result = json.loads(line)
                    cluster_id = result['cluster']
                    if cluster_id in self.excluded_clusters:
                        continue  # Skip excluded clusters
                    # Store index for the cluster
                    cluster_indices[cluster_id].append(idx)
        # Step 2: Randomly sample `examples_per_cluster` indices for each cluster
        for cluster_id, indices in cluster_indices.items():
            # Ensure we sample up to `examples_per_cluster` from the available indices
            sampled_indices[cluster_id] = set(random.sample(indices, min(self.config.examples_per_cluster, len(indices))))
        return sampled_indices

    def create_minipile(self) -> None:
        if not self.excluded_clusters:
            raise ValueError("Call exclude_clusters() before creating MiniPile.")
        
        sampled_indices = self._get_cluster_sample_indices()
        total_examples, kept_examples, shard_idx = 0, 0, 0
        output_batch = []

        # Process embeddings shard by shard
        with tqdm(total=None, desc="Processing embeddings") as pbar:
            for embd_file in sorted(self.config.embd_dir.glob("shard_*.parquet")):
                # Load embeddings
                table = pq.read_table(embd_file)
                num_rows = len(table)

                # Load clustering results for this shard
                cluster_results = []
                cluster_file = self.config.cluster_dir / f"clustering_results_{embd_file.stem}.jsonl"
                with open(cluster_file, 'r') as f:
                    for line in f:
                        result = json.loads(line)
                        cluster_results.append((result['cluster'], result['distance']))

                if len(cluster_results) != num_rows:
                    raise ValueError(f"Mismatch between embeddings ({num_rows}) and cluster results ({len(cluster_results)}) for {embd_file.name}.")

                # Filter embeddings
                keep_indices = []
                for i, (cluster_id, distance) in enumerate(cluster_results):
                    if cluster_id not in self.excluded_clusters and i in sampled_indices.get(cluster_id, set()):
                        keep_indices.append(i)

                kept_table = table.take(keep_indices)
                output_batch.append(kept_table)
                total_examples += num_rows
                kept_examples += len(kept_table)

                # Write to output shard if batch size is met
                if sum(len(t) for t in output_batch) >= self.config.output_shard_size:
                    self._write_shard(output_batch, shard_idx)
                    output_batch = []
                    shard_idx += 1

                pbar.update(num_rows)

        # Write remaining examples
        if output_batch:
            self._write_shard(output_batch, shard_idx)

        # Save metadata
        metadata = {
            "total_examples": total_examples,
            "kept_examples": kept_examples,
            "percentage_kept": (kept_examples / total_examples) * 100 if total_examples > 0 else 0,
            "output_shards": shard_idx + 1,
            "excluded_clusters": sorted(list(self.excluded_clusters))
        }
        with open(self.config.output_dir / "minipile_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMiniPile created with {kept_examples:,} examples ({metadata['percentage_kept']:.2f}%) across {shard_idx + 1} shards.")

    def _write_shard(self, tables: list[pa.Table], shard_idx: int) -> None:
        combined_table = pa.concat_tables(tables)
        shard_path = self.config.output_dir / f"shard_{shard_idx:09d}.parquet"
        pq.write_table(combined_table, shard_path)
        print(f"Shard {shard_idx} written: {len(combined_table):,} examples.")

if __name__ == "__main__":
    config = FilterConfig()
    filter = MiniPileFilter(config)

    # Manual exclusion of clusters after inspection
    clusters_to_exclude = []  # Replace with actual cluster IDs
    filter.exclude_clusters(clusters_to_exclude)

    # Create MiniPile
    filter.create_minipile()