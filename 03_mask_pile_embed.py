import json
import queue
import threading
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Dict, List
from collections import defaultdict

# This doesn't work and it was left here solely for reference. This established a lot of the structure that I used for the actual implementation.
# I tinkered with this masking step idea for a while, ultimately trying to find a most appropriate way to distill the MiniPile dataset.
# I moved on to build the working 03_distill_pile_embed.py

@dataclass
class FilterConfig:
    base_dir: Path = Path("/vol/tmp/koppelmm")
    batch_size: int = 16384                                  # Batch size for processing embeddings, I keep it the same as for clustering
    num_clusters: int = 220                                  # Number of clusters used for clustering The Pile Deduplicated (Embd)
    num_clusters_to_exclude: int = 38                        # Number of clusters to exclude (as per paper, referenced here for sanity-checks)
    examples_per_cluster: int = int(1010500 / num_clusters)  # ~1M total retained examples (exact number from HuggingFace) (~4593 data points per cluster)
    
    def __post_init__(self):
        self.embd_dir = self.base_dir / "Pile_Deduplicated_Embd"    # Dataset w/ embeddings
        self.cluster_dir = self.base_dir / "MiniPile_BatchKMeans"   # Clustering results
        self.output_dir = self.base_dir / "MiniPile_Self_Meta"      # Target for mask and metadata
        self.output_dir.mkdir(exist_ok=True, parents=True)

class AsyncMaskWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.write_queue = queue.Queue(maxsize=4)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        self.current_chunk = 0
    
    def _writer_loop(self):
        while True:
            data = self.write_queue.get()
            if data is None:
                break
            mask_chunk = data
            chunk_path = self.output_dir / f"mask_chunk_{self.current_chunk:09d}.npy"
            np.save(chunk_path, mask_chunk)
            self.current_chunk += 1
            
            # Memory cleanup
            del mask_chunk
            del data
            del chunk_path
            
            self.write_queue.task_done()
    
    def write(self, mask_chunk: np.ndarray):
        self.write_queue.put(mask_chunk)
    
    def finish(self):
        self.write_queue.put(None)
        self.writer_thread.join()

class MiniPileFilter:
    def __init__(self, config: FilterConfig):
        self.config = config
        self.excluded_clusters: Set[int] = set() # Set of excluded cluster IDs (38 clusters)
        self._load_cluster_info()
    
    def _load_cluster_info(self) -> None:
        with open(self.config.cluster_dir / "cluster_info_for_inspection.json", "r") as f:
            self.cluster_info = json.load(f)
    
    def display_cluster_examples(self, cluster_id: int) -> None:
        # Display (per cluster) the 5 closest and farthest examples for inspection
        info = self.cluster_info[str(cluster_id)]
        print(f"\nCluster {cluster_id}:")
        print(f"\tTotal examples: {info['total_examples']}")
        print(f"\tAverage distance: {info['average_distance']:.4f}")
        # Display just the first sym_count characters per example for readability
        sym_count = 512
        print("\n\tClosest Examples:")
        for i, ex in enumerate(info['closest'], 1):
            print(f"\n\t\t{i}. Distance: {ex['distance']:.4f}")
            print(f"\t\t{ex['text'][:sym_count] + "..." if len(ex['text']) > sym_count else ex['text']}")
        print("\n\tFarthest Examples:")
        for i, ex in enumerate(info['farthest'], 1):
            print(f"\n\t\t{i}. Distance: {ex['distance']:.4f}")
            print(f"\t\t{ex['text'][:sym_count] + "..." if len(ex['text']) > sym_count else ex['text']}")
    
    def exclude_clusters(self, cluster_ids: List[int]) -> None:
        # Sanity-check and persist a human's choice for excluded clusters
        self.excluded_clusters = set(cluster_ids)
        if self.excluded_clusters and len(self.excluded_clusters) != self.config.num_clusters_to_exclude:
            raise ValueError(f"Must exclude exactly {self.config.num_clusters_to_exclude} (unique) clusters to adhere to the paper's specifications")
        # Persist our excluded cluster choices
        with open(self.config.output_dir / "excluded_clusters.json", "w") as f:
            json.dump({"excluded_clusters": sorted(list(self.excluded_clusters))}, f, indent=2)
    
    def _get_cluster_thresholds(self) -> Dict[int, float]:
        # Calculate distance thresholds for each non-excluded cluster
        results_dir = self.config.cluster_dir / "clustering_results"
        # Dict to hold thresholds for each cluster
        thresholds = {}
        # Get all distances for each cluster
        cluster_distances = defaultdict(list)
        
        for chunk_file in tqdm(sorted(results_dir.glob("cluster_results_chunk_*.jsonl")), desc="Accumulating cluster distances"):
            with open(chunk_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    if result['cluster'] not in self.excluded_clusters:
                        cluster_distances[result['cluster']].append(result['distance'])
        
        # Calculate threshold pereach cluster, retaining examples_per_cluster examples
        for cluster_id, distances in tqdm(cluster_distances.items(), desc="Calculating thresholds"):
            sorted_distances = sorted(distances)
            threshold_idx = min(self.config.examples_per_cluster, len(sorted_distances) - 1)
            thresholds[cluster_id] = sorted_distances[threshold_idx]
        
        return thresholds
    
    def create_chunked_mask(self) -> None:
        # Create mask by streaming through embedded Pile Dedup set
        if not self.excluded_clusters:
            # Stop for first iteration to ensure we can human-inspect the clusters
            raise ValueError("Call exclude_clusters() prior to mask creation")
        
        writer = AsyncMaskWriter(self.config.output_dir)
        total_examples, kept_examples = 0, 0

        # Get thresholds for remaining clusters
        cluster_thresholds = self._get_cluster_thresholds()

        # Process each embedding shard        
        with tqdm(total=None, desc="Processing shards") as pbar:
            # Load clustering results
            results_dir = self.config.cluster_dir / "clustering_results"
            for chunk_file in sorted(results_dir.glob("cluster_results_chunk_*.jsonl")):
                # Load results from chunk file
                cluster_results = []
                with open(chunk_file, 'r') as f:
                    for line in f:
                        result = json.loads(line)
                        cluster_results.append((result['cluster'], result['distance']))
                
                # Create mask for this chunk
                chunk_mask_list = []
                for cluster_id, distance in cluster_results:
                    # Flat-out exclude if cluster is in excluded set
                    if cluster_id in self.excluded_clusters:
                        chunk_mask_list.append(False)
                        continue
                    threshold = cluster_thresholds[cluster_id]
                    keep = distance <= threshold # Keep if distance to centroid is below threshold
                    chunk_mask_list.append(keep)
                    if keep:
                        kept_examples += 1
                
                chunk_mask = np.array(chunk_mask_list, dtype=bool)
                chunk_mask_len = len(chunk_mask)
                total_examples += chunk_mask_len
                
                # Write mask chunk
                writer.write(chunk_mask)
                pbar.update(chunk_mask_len)

                # Memory cleanup
                del chunk_mask
                del chunk_mask_list
                del cluster_results
        
        # Close up shop
        writer.finish()
        
        metadata = {"total_examples": total_examples,
                    "kept_examples": kept_examples,
                    "percentage_kept": (kept_examples / total_examples) * 100 if total_examples > 0 else 0,
                    "excluded_clusters": sorted(list(self.excluded_clusters))}
        
        # I want this transparent, so save metadata from mask creation
        with open(self.config.output_dir / "minipile_mask_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("\nMask Creation Stats:")
        print(f"\tTotal examples processed: {total_examples:,}")
        print(f"\tExamples kept: {kept_examples:,}")
        print(f"\tPercentage kept: {metadata['percentage_kept']:.3f}%")

if __name__ == "__main__":
    config = FilterConfig()
    filter = MiniPileFilter(config)
    
    # Display clusters for inspection
    for cluster_id in range(config.num_clusters):
        filter.display_cluster_examples(cluster_id)

    # TODO: Fill list with 38 unique cluster ids (int, start at 0) via manual inspection
    clusters_to_exclude = []
    filter.exclude_clusters(clusters_to_exclude)
    
    # Create mask for MiniPile dataset based on 
    # a) Chosen clusters to exclude, and
    # b) Remaining clusters' examples meeting the thresholds for examples_per_cluster and distance.
    filter.create_chunked_mask()
    
    print("\n[!] MiniPile mask created")

# Run this to create the mask for MiniPile
# This doesn't copy or move data or anything, it just creates the mask and metadata
# To make this have an effect and to effectively 'cut-out' the MiniPile dataset, use something like this (just a general idea, not a working example, I'm tired):

# I had this idea for successive training jobs:
# 
#def load_with_mask(embd_shard_idx):
#    # Load a shard (TODO: This isn't streaming the dataset yet)
#    embd_data = load_dataset("parquet", 
#                             data_files=f"shard_{embd_shard_idx:09d}.parquet", 
#                             split="train")
#    # Load a mask chunk (TODO: Make sure correct indices are used, even when in multi-GPU setup)
#    mask = np.load(f"mask_chunk_{embd_shard_idx:09d}.npy")
#    # Filter data
#    return embd_data.select(np.where(mask)[0])
# 
# But honestly, while I think this could work, the overhead this would create for training is such a nightmare that I don't want it.
# I think biting the bullet and creating a new dataset is the way to go. I'll do that instead. But hey, considering options, right?

# Writing a seperate 03_distill_pile.py script to create the MiniPile dataset is prob best for SoC and readability.