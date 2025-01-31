import json
import time
import gc
import numpy as np
import multiprocessing
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# Literally no idea what I was thinking. I was forcing HDBSCAN as a hypothetical in here.
# HDBSCAN was added to sklearn in 1.3 https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
# But this is a mess. The impossible-to-batch nature of HDBSCAN is known is makes all of this pointless if there's no way to batch it.
# Didn't find a good batching approach.

base_path = Path("/vol/tmp/koppelmm")
embd_dir = base_path / "Pile_Deduplicated_Embd"
cluster_dir = base_path / "MiniPile_HDBSCAN"
cluster_dir.mkdir(exist_ok=True)

# HDBSCAN params
batch_size = 1_000_000
min_cluster_size = 1_000  # Minimum cluster size to avoid noise
min_samples = 2_000       # Minimum number of samples in a neighborhood for core point
cluster_selection_epsilon = 0.05  # Allow some flexibility in cluster boundary
n_jobs = multiprocessing.cpu_count() // 2  # Utilize half of available CPU cores

cluster_results_path = cluster_dir / "clustering_results"
cluster_results_path.mkdir(exist_ok=True)

class ChunkedResultWriter:
    def __init__(self, output_dir: Path, chunk_size: int = 1_000_000, prefix: str = "cluster_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chunk_size = chunk_size
        self.prefix = prefix
        self.current_chunk = self._get_next_chunk_index()
        self.entries_in_current_chunk = 0
        self.current_file = None
        self._open_new_chunk()
    
    def _get_next_chunk_index(self):
        existing_shards = list(self.output_dir.glob(f"{self.prefix}_chunk_*.jsonl"))
        return int(len(existing_shards))

    def _open_new_chunk(self):
        if self.current_file is not None:
            self.current_file.close()
        
        chunk_path = self.output_dir / f"{self.prefix}_chunk_{self.current_chunk:09d}.jsonl"
        self.current_file = open(chunk_path, 'w', buffering=1)
    
    def write_result(self, result):
        try:
            self.current_file.write(json.dumps(result) + '\n')
            self.entries_in_current_chunk += 1
            
            if self.entries_in_current_chunk >= self.chunk_size:
                self.current_chunk += 1
                self.entries_in_current_chunk = 0
                self._open_new_chunk()
            return True
        except Exception as e:
            print(f"Error writing result: {e}")
            return False
    
    def close(self):
        if self.current_file is not None:
            self.current_file.close()
        
        metadata = {"total_chunks": self.current_chunk + 1,
                    "chunk_size": self.chunk_size,
                    "prefix": self.prefix,
                    "total_entries": (self.current_chunk * self.chunk_size) + self.entries_in_current_chunk}
        
        with open(self.output_dir / f"{self.prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

def process_embeddings():
    shards = sorted(list(embd_dir.glob("shard_*.parquet")))
    writer = ChunkedResultWriter(output_dir=cluster_results_path, chunk_size=1_000_000, prefix="cluster_results")
    
    total_processed = 0
    existing_shard_count = int(len(list(cluster_results_path.glob("cluster_results_chunk_*.jsonl"))))
    skip_items_count = (existing_shard_count - 1) * writer.chunk_size if existing_shard_count > 0 else 0
    
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, 
                        min_samples=min_samples,
                        cluster_selection_epsilon=cluster_selection_epsilon,
                        metric='cosine',
                        n_jobs=n_jobs)

    if skip_items_count > 0:
        # Complex skipping logic similar to original script
        entries_per_shard_52 = 1_048_576
        entries_per_shard_post = 524_288
        large_shard_count = 52

        if skip_items_count < large_shard_count * entries_per_shard_52:
            shard_index = skip_items_count // entries_per_shard_52
            local_skip = skip_items_count % entries_per_shard_52
        else:
            remaining_skip = skip_items_count - (large_shard_count * entries_per_shard_52)
            shard_index = large_shard_count + (remaining_skip // entries_per_shard_post)
            local_skip = remaining_skip % entries_per_shard_post

        print(f'Skipping to shard idx: {shard_index}, local idx: {local_skip}')
        data_files = shards[shard_index:]
        dataset = load_dataset("parquet", data_files={"train": [str(file) for file in data_files]}, split="train", streaming=True)
        dataset = dataset.skip(local_skip)
        total_processed = skip_items_count
    else:
        dataset = load_dataset("parquet", data_files=str(embd_dir / "*.parquet"), split="train", streaming=True)
    
    all_mini_clusters = []
    
    with tqdm(total=None, desc="Two-Phase Clustering") as pbar:
        try:
            for batch in dataset.iter(batch_size=batch_size):
                embeddings = np.array(batch['embedding'])
                
                # Phase 1: Create mini-clusters
                mini_clusters, labels = create_mini_clusters(embeddings, clusterer)
                all_mini_clusters.extend(mini_clusters)
                
                # Write temporary results
                for idx, label in enumerate(labels):
                    result = {'idx': total_processed + idx, 'temp_cluster': int(label)}
                    writer.write_result(result)
                
                total_processed += len(embeddings)
                pbar.update(len(embeddings))
                
                del embeddings
                gc.collect()
            
            # Phase 2: Merge mini-clusters
            print("[~] Merging mini-clusters")
            merged_clusters = merge_mini_clusters(all_mini_clusters, distance_threshold=0.1)
            
            # Create a mapping from original mini-cluster index to final cluster label
            cluster_mapping = {}
            for final_label, cluster in enumerate(merged_clusters):
                for mini_cluster_idx in cluster:
                    cluster_mapping[mini_cluster_idx] = final_label
            
            # Update cluster assignments
            print("[~] Updating cluster assignments")
            for chunk_file in sorted(cluster_results_path.glob("cluster_results_chunk_*.jsonl")):
                with open(chunk_file, 'r') as f:
                    lines = f.readlines()
                
                updated_results = []
                for line in lines:
                    result = json.loads(line)
                    temp_cluster = result['temp_cluster']
                    if temp_cluster != -1 and temp_cluster < len(all_mini_clusters):
                        result['cluster'] = cluster_mapping.get(temp_cluster, -1)
                    else:
                        result['cluster'] = -1
                    del result['temp_cluster']
                    updated_results.append(result)
                
                with open(chunk_file, 'w') as f:
                    for result in updated_results:
                        f.write(json.dumps(result) + '\n')
            
        finally:
            writer.close()

    print("[+] Two-phase clustering completed.")
    
def create_mini_clusters(embeddings, clusterer):
    labels = clusterer.fit_predict(embeddings)
    unique_labels = np.unique(labels)
    mini_clusters = []
    for label in unique_labels:
        if label != -1:  # Exclude noise points
            cluster_points = embeddings[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            mini_clusters.append({
                'centroid': centroid,
                'size': len(cluster_points),
                'label': label
            })
    return mini_clusters, labels

def merge_mini_clusters(mini_clusters, distance_threshold):
    centroids = np.array([mc['centroid'] for mc in mini_clusters])
    nn = NearestNeighbors(n_neighbors=2, metric='cosine')
    nn.fit(centroids)
    distances, indices = nn.kneighbors(centroids)
    
    merged_clusters = {}
    for i, (distance, idx) in enumerate(zip(distances[:, 1], indices[:, 1])):
        if distance < distance_threshold:
            smaller_idx = min(i, idx)
            larger_idx = max(i, idx)
            if smaller_idx not in merged_clusters:
                merged_clusters[smaller_idx] = {smaller_idx, larger_idx}
            else:
                merged_clusters[smaller_idx].add(larger_idx)
    
    final_clusters = []
    processed = set()
    for key, cluster in merged_clusters.items():
        if key not in processed:
            new_cluster = set()
            for idx in cluster:
                new_cluster.update(merged_clusters.get(idx, {idx}))
            final_clusters.append(list(new_cluster))
            processed.update(new_cluster)
    
    return final_clusters

if __name__ == "__main__":
    process_embeddings()