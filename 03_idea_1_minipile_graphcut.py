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
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

## Idea 1:
#   Graph Cut may help selecting a subset balancing representativeness and diversity
#   I think of it as a rather principled way to optimize the selection process, potentially improving upon solely random sampling
#   This idea also adjusts the number of documents selected from each cluster based on the cluster's size
#   Due to inherently more computational complexity, we can't run this on all cluster points, 
#   but at max(cluster_size, n * examples_per_cluster) (say, n=8) points to distill from that the total examples_per_cluster points we add to the MiniPile
# Chose that approach based on requirements for speed and efficiency, whie still introducing a measure for diversity 
# (based on results from "DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning")
# Note however (that I'm aware) that this doesn't explicitly account for example difficulty from a language modeling perspective.

@dataclass
class FilterConfig:
    base_dir: Path = Path("/vol/tmp/koppelmm")
    batch_size: int = 16384
    num_clusters: int = 220
    num_clusters_to_exclude: int = 38
    examples_per_cluster: int = int(1010500 / (num_clusters - num_clusters_to_exclude))
    output_shard_size: int = 100_000
    lambda_diversity: float = 2.0
    graph_cut_multiplier: int = 8  # Multiplier for Graph Cut sample size
    
    def __post_init__(self):
        self.embd_dir = self.base_dir / "Pile_Deduplicated_Embd"
        self.cluster_dir = self.base_dir / "MiniPile_BatchKMeans"
        self.output_dir = self.base_dir / "MiniPile_GraphCut"
        self.output_dir.mkdir(exist_ok=True, parents=True)

class MiniPileFilter:
    def __init__(self, config: FilterConfig):
        self.config = config
        self.excluded_clusters: Set[int] = set()
        self._load_cluster_info()
    
    def _load_cluster_info(self) -> None:
        with open(self.config.cluster_dir / "cluster_info_for_inspection.json", "r") as f:
            self.cluster_info = json.load(f)
    
    def exclude_clusters(self, cluster_ids: List[int]) -> None:
        self.excluded_clusters = set(cluster_ids)
        if len(self.excluded_clusters) != self.config.num_clusters_to_exclude:
            print(f"[!] Must exclude exactly {self.config.num_clusters_to_exclude} clusters. You provided {len(self.excluded_clusters)} clusters.")
        with open(self.config.output_dir / "excluded_clusters.json", "w") as f:
            json.dump({"excluded_clusters": sorted(list(self.excluded_clusters))}, f, indent=2)
    
    def _graph_cut_selection(self, embeddings: np.ndarray, n_select: int) -> List[int]:
        normalized_embeddings = normalize(embeddings)
        selected = []
        remaining = list(range(len(embeddings)))
        
        for _ in range(n_select):
            if not remaining:
                break
            
            remaining_embeddings = normalized_embeddings[remaining]
            selected_embeddings = normalized_embeddings[selected] if selected else np.empty((0, normalized_embeddings.shape[1]))
            
            representativeness = np.sum(1 - cdist(remaining_embeddings, normalized_embeddings, metric='cosine'), axis=1)
            diversity = self.config.lambda_diversity * np.sum(1 - cdist(remaining_embeddings, selected_embeddings, metric='cosine'), axis=1)
            
            scores = representativeness - diversity
            best_idx = remaining[np.argmax(scores)]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected

    def _get_cluster_sample_indices(self) -> Dict[int, Set[int]]:
        results_dir = self.config.cluster_dir / "clustering_results"
        cluster_indices = defaultdict(list)
        sampled_indices = {}
        
        for chunk_file in tqdm(sorted(results_dir.glob("cluster_results_chunk_*.jsonl")), desc="Aggregating cluster indices"):
            with open(chunk_file, 'r') as f:
                for idx, line in enumerate(f):
                    result = json.loads(line)
                    cluster_id = result['cluster']
                    if cluster_id not in self.excluded_clusters:
                        cluster_indices[cluster_id].append(idx)
        
        for cluster_id, indices in tqdm(cluster_indices.items(), desc="Applying Graph Cut"):
            graph_cut_sample_size = min(len(indices), self.config.graph_cut_multiplier * self.config.examples_per_cluster)
            graph_cut_indices = random.sample(indices, graph_cut_sample_size)
            embeddings = self._load_embeddings(graph_cut_indices)
            selected = self._graph_cut_selection(embeddings, min(self.config.examples_per_cluster, graph_cut_sample_size))
            sampled_indices[cluster_id] = set(graph_cut_indices[i] for i in selected)
        
        return sampled_indices

    def _load_embeddings(self, indices: List[int]) -> np.ndarray:
        embeddings = []
        for embd_file in sorted(self.config.embd_dir.glob("shard_*.parquet")):
            table = pq.read_table(embd_file, columns=['embedding'])
            shard_indices = [i for i in indices if i < len(table)]
            if shard_indices:
                embeddings.extend(table.take(shard_indices).to_pandas()['embedding'].tolist())
            indices = [i - len(table) for i in indices if i >= len(table)]
            if not indices:
                break
        return np.array(embeddings)

    def create_minipile(self) -> None:
        if not self.excluded_clusters:
            raise ValueError("Call exclude_clusters() before creating MiniPile.")
        
        sampled_indices = self._get_cluster_sample_indices()
        total_examples, kept_examples, shard_idx = 0, 0, 0
        output_batch = []

        with tqdm(total=None, desc="Processing embeddings") as pbar:
            for embd_file in sorted(self.config.embd_dir.glob("shard_*.parquet")):
                table = pq.read_table(embd_file)
                num_rows = len(table)

                cluster_results = []
                cluster_file = self.config.cluster_dir / f"clustering_results_{embd_file.stem}.jsonl"
                with open(cluster_file, 'r') as f:
                    for line in f:
                        result = json.loads(line)
                        cluster_results.append((result['cluster'], result['distance']))

                if len(cluster_results) != num_rows:
                    raise ValueError(f"Mismatch between embeddings ({num_rows}) and cluster results ({len(cluster_results)}) for {embd_file.name}.")

                keep_indices = [i for i, (cluster_id, _) in enumerate(cluster_results) 
                                if cluster_id not in self.excluded_clusters and 
                                i in sampled_indices.get(cluster_id, set())]

                kept_table = table.take(keep_indices)
                output_batch.append(kept_table)
                total_examples += num_rows
                kept_examples += len(kept_table)

                if sum(len(t) for t in output_batch) >= self.config.output_shard_size:
                    self._write_shard(output_batch, shard_idx)
                    output_batch = []
                    shard_idx += 1

                pbar.update(num_rows)

        if output_batch:
            self._write_shard(output_batch, shard_idx)

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
    clusters_to_exclude = []  # Replace with actual cluster IDs
    filter.exclude_clusters(clusters_to_exclude)
    filter.create_minipile()