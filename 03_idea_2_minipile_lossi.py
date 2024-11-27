import json
import numpy as np
from pathlib import Path
from typing import Set, Dict, List, Tuple
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import defaultdict
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

## Idea 2:
#   Combine the best of random sampling and loss-based sampling, opening opportunity to create a dataset that is both representative and challenging
#   We still randomly sample, but a loss_sample_multiplier * examples_per_cluster number of examples are selected randomly now
#   From these, random_sample_ratio * examples_per_cluster examples are kept outright,
#   (1 - random_sample_ratio) * examples_per_cluster examples are selected based on loss, we perform inference on a half-way trained Pythia 160M model
#   The top (1 - random_sample_ratio) * examples_per_cluster examples are kept from the (loss_sample_multiplier * examples_per_cluster - pre_selected_outright) examples
#   I retain random sampling as it help mitigate making the total distillation too challenging, and still,
#   loss-based sampling helps to make the dataset challenging instead of totally uniformly random
# Related to the paper "Coverage-Centric Scoreset Selection For High Purning Rates"

@dataclass
class FilterConfig:
    base_dir: Path = Path("/vol/tmp/koppelmm")
    batch_size: int = 16384
    num_clusters: int = 220
    num_clusters_to_exclude: int = 38
    examples_per_cluster: int = int(1010500 / (num_clusters - num_clusters_to_exclude))
    output_shard_size: int = 100_000
    loss_sample_multiplier: int = 4
    random_sample_ratio: float = 0.4
    
    def __post_init__(self):
        self.embd_dir = self.base_dir / "Pile_Deduplicated_Embd"
        self.cluster_dir = self.base_dir / "MiniPile_BatchKMeans"
        self.output_dir = self.base_dir / "MiniPile_HybridSampling"
        self.output_dir.mkdir(exist_ok=True, parents=True)

class MiniPileFilter:
    def __init__(self, config: FilterConfig):
        self.config = config
        self.excluded_clusters: Set[int] = set()
        self._load_cluster_info()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").to(self.device)
        self.model.eval()

    def _load_cluster_info(self) -> None:
        with open(self.config.cluster_dir / "cluster_info_for_inspection.json", "r") as f:
            self.cluster_info = json.load(f)
    
    def exclude_clusters(self, cluster_ids: List[int]) -> None:
        self.excluded_clusters = set(cluster_ids)
        if len(self.excluded_clusters) != self.config.num_clusters_to_exclude:
            print(f"[!] Must exclude exactly {self.config.num_clusters_to_exclude} clusters. You provided {len(self.excluded_clusters)} clusters.")
        with open(self.config.output_dir / "excluded_clusters.json", "w") as f:
            json.dump({"excluded_clusters": sorted(list(self.excluded_clusters))}, f, indent=2)

    def _compute_loss(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()

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
        
        for cluster_id, indices in tqdm(cluster_indices.items(), desc="Applying Hybrid Sampling"):
            total_samples = min(len(indices), self.config.examples_per_cluster)
            random_samples = int(total_samples * self.config.random_sample_ratio)
            loss_based_samples = total_samples - random_samples

            # Random sampling
            random_selected = set(random.sample(indices, random_samples))
            
            # Loss-based sampling
            remaining_indices = list(set(indices) - random_selected)
            loss_sample_size = min(len(remaining_indices), self.config.loss_sample_multiplier * loss_based_samples)
            loss_candidates = random.sample(remaining_indices, loss_sample_size)
            
            losses = []
            for idx in loss_candidates:
                text = self._load_text(idx)
                loss = self._compute_loss(text)
                losses.append((idx, loss))
            losses.sort(key=lambda x: x[1], reverse=True)
            loss_selected = set(x[0] for x in losses[:loss_based_samples])

            sampled_indices[cluster_id] = random_selected.union(loss_selected)
        
        return sampled_indices

    def _load_text(self, index: int) -> str:
        # Implement this method to load the text for a given index
        # This is a placeholder implementation
        return f"Sample text for index {index}"

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
            "excluded_clusters": sorted(list(self.excluded_clusters)),
            "random_sample_ratio": self.config.random_sample_ratio,
            "loss_sample_multiplier": self.config.loss_sample_multiplier
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