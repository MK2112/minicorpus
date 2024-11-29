import json
import time
import gc
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_distances
from datasets import load_dataset
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

base_path = Path("/vol/tmp/koppelmm")
embd_dir = base_path / "Pile_Deduplicated_Embd" # This is where the embeddings are stored/written to (create "End_Here.txt" here to signal end)
cluster_dir = base_path / "MiniPile_BatchKMeans"
cluster_dir.mkdir(exist_ok=True)

k_clusters = 220    # As per paper
batch_size = 16384  # As per paper
n_init = 3          # Default, nothing else is specified

class CosineMiniBatchKMeans(MiniBatchKMeans):
    # Stupidly many parameters, but necessary (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
    def __init__(self, n_clusters=8, *, init='k-means++', max_iter=100, batch_size=1024, 
                 verbose=0, compute_labels=True, random_state=None, tol=0.0, 
                 max_no_improvement=10, init_size=None, n_init='auto', reassignment_ratio=0.01):
        super().__init__(n_clusters=n_clusters, batch_size=batch_size,
                         init=init, n_init=n_init, max_iter=max_iter,
                         verbose=verbose, random_state=random_state,
                         tol=tol, max_no_improvement=max_no_improvement,
                         reassignment_ratio=reassignment_ratio, 
                         compute_labels=compute_labels, init_size=init_size)
        self._n_threads = 32

    def _transform(self, X):
        # Prob most lowkey way to enforce using cosine distance
        return cosine_distances(X, self.cluster_centers_)

    def _mini_batch_step(self, X, sample_weight, x_squared_norms, random_reassign=False, n_threads=32):
        # Plainly call original method for batch processing
        super()._mini_batch_step(X, sample_weight, x_squared_norms, random_reassign, n_threads)
        # Normalize the centroids, remain on unit hypersphere for interpretability
        self.cluster_centers_ = self.cluster_centers_ / np.linalg.norm(self.cluster_centers_, axis=1, keepdims=True)

batchified_kmeans = CosineMiniBatchKMeans(n_clusters=k_clusters, batch_size=batch_size, init='k-means++', 
                                          n_init=n_init, random_state=42)

class ChunkedResultWriter:
    def __init__(self, output_dir: Path, chunk_size: int = 1_000_000, prefix: str = "clustering_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.chunk_size = chunk_size
        self.prefix = prefix
        self.current_chunk = 0
        self.entries_in_current_chunk = 0
        self.current_file = None
        self.last_written_idx = -1
        self._open_new_chunk()
    
    def _open_new_chunk(self):
        if self.current_file is not None:
            self.current_file.close()
            
        chunk_path = self.output_dir / f"{self.prefix}_chunk_{self.current_chunk:09d}.jsonl"
        self.current_file = open(chunk_path, 'w', buffering=1)
    
    def write_result(self, result: Dict[str, Any]) -> bool:
        if result['idx'] != self.last_written_idx + 1:
            raise ValueError("Out of order write detected.")

        try:
            self.current_file.write(json.dumps(result) + '\n')
            self.entries_in_current_chunk += 1
            self.last_written_idx = result['idx']
            
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
            
        # Write metadata about all the chunks
        metadata = {
            "total_chunks": self.current_chunk + 1,
            "chunk_size": self.chunk_size,
            "prefix": self.prefix,
            "total_entries": (self.current_chunk * self.chunk_size) + self.entries_in_current_chunk
        }
        
        with open(self.output_dir / f"{self.prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


# I've seen that being used by Hugging Face for storing results 
# (https://www.atatus.com/glossary/jsonl/) seems to save some disk space by structure
cluster_info_path = cluster_dir / "cluster_info_for_inspection.json"
clustering_results_path = cluster_dir / "clustering_results.jsonl"
cluster_centers_path = cluster_dir / "cluster_centers.npy"

# Get cos distances using latest centroids
def compute_distances(embeddings, centroids, labels):
    return 1 - np.sum(embeddings * centroids[labels], axis=1)

# Get closest and farthest examples for a cluster
def get_extreme_examples(embeddings, labels, centroids, texts, n=5):
    distances = cosine_distances(embeddings, centroids[labels]).diagonal()
    sorted_indices = np.argsort(distances)
    closest_indices = sorted_indices[:n]   # First n
    farthest_indices = sorted_indices[-n:] # Furthest n, thank god for slicing
    return ([{"text": texts[idx], "distance": distances[idx]} for idx in closest_indices],
            [{"text": texts[idx], "distance": distances[idx]} for idx in farthest_indices])

# Cluster tracking infos
cluster_info = {
    i: {
        'closest': [],
        'farthest': [],
        'total_examples': 0,
        'average_distance': 0.0,
        'sum_distance': 0.0
    } for i in range(k_clusters)
}

last_filename = None         # Load files with indices newer than this only
checkpoint_shard_counter = 0 # Shard counter for checkpointing centroids

def monitor_and_fit():
    global last_filename
    global checkpoint_shard_counter
    
    while True:
        # Files are of format: shard_000000000.parquet
        # Check if there's a new file
        shards = sorted(list(embd_dir.glob("shard_*.parquet")))
        end_signal_given = (embd_dir / "End_Here.txt").exists()
        
        # Check if a model has already been trained, load it and skip training
        if cluster_centers_path.exists():
            print("Loading existing cluster centers...")
            batchified_kmeans.cluster_centers_ = np.load(cluster_centers_path)
            break

        # Remove all files that have already been processed (< last_filename number in name)
        if last_filename:
            shards = [shard for shard in shards if int(shard.stem.split("_")[1]) > int(last_filename.stem.split("_")[1])]

        if len(shards) == 0 and not end_signal_given:
            # Wait for 10 minutes before checking again, greedy but it's fine
            print("Idle. Waiting for new files...")
            time.sleep(600)
            continue # Skip the rest of the loop

        # Check for new files being written
        if shards:
            # Get the most recent shard
            last_shard = sorted(shards)[-1]
            last_modified_time = last_shard.stat().st_mtime
            
            # Check if the file was modified recently (e.g., within the last 6.667 minutes)
            # Back off to have potential writing processes conclude
            if time.time() - last_modified_time < 400: 
                print(f"Detected recent modification ({time.time() - last_modified_time}). Backing off for another 5 minutes...")
                time.sleep(300)  # Wait for an additional 5 minutes
                continue

        last_filename = sorted(shards)[-1] if not end_signal_given else ""

        # Process each individual parquet file
        for shard in shards:
            shardaset = load_dataset("parquet", data_files=str(shard), split="train", streaming=True)
            # Process shardaset in batches according to paper
            with tqdm(total=None, desc="Processing Batches") as pbar:
                for batch in shardaset.iter(batch_size=batch_size):
                    embeddings = np.array(batch['embedding'])
                    texts = batch['text']
                    # Batched fitting
                    batchified_kmeans.partial_fit(embeddings)
                    pbar.update(len(texts))
                    del embeddings, texts
                    gc.collect()
            checkpoint_shard_counter += 1

            del shardaset
            gc.collect()

            # Save the model as checkpoint every 128 shards
            if checkpoint_shard_counter % 4 == 0:
                np.save(cluster_dir / f"cluster_centers_shard_{checkpoint_shard_counter}.npy", batchified_kmeans.cluster_centers_)

        # Place this here to process residual parquet files before
        if end_signal_given:
            print("End signal found. Packing up clustering...")
            break

    # After processing all files, save the batchified_kmeans model
    # cluster_centers.npy
    np.save(cluster_centers_path, batchified_kmeans.cluster_centers_)
    # Continue to predict clusters, save results thereof
    finalize_clustering()

def finalize_clustering():
    # Now load all embeddings from all parquet files to cluster them
    dataset = load_dataset("parquet", data_files=str(embd_dir / "*.parquet"), split="train", streaming=True)
    total_processed = 0 # Track number of processed examples
    cluster_info_temp = defaultdict(lambda: {'closest': [], 'farthest': [], 'total_examples': 0, 'sum_distance': 0.0})

    writer = ChunkedResultWriter(output_dir=cluster_dir / "clustering_results", chunk_size=1_000_000, prefix="cluster_results")

    with tqdm(total=None, desc="Final Prediction") as pbar:
        try:
            for batch in dataset.iter(batch_size=batch_size):
                embeddings = np.array(batch['embedding'])
                texts = batch['text']

                # Predict clusters and compute distances
                labels = batchified_kmeans.predict(embeddings)
                distances = compute_distances(embeddings, batchified_kmeans.cluster_centers_, labels)

                # Write clustering results for each example
                for idx, (text, label, distance) in enumerate(zip(texts, labels, distances)):
                    result = {
                        'idx': total_processed + idx,
                        'cluster': int(label),
                        'distance': float(distance)
                    }
                    
                    if writer.write_result(result):
                        del result

                    # Update cluster info
                    cluster = int(label)
                    cluster_info_temp[cluster]['total_examples'] += 1
                    cluster_info_temp[cluster]['sum_distance'] += distance
                    cluster_info_temp[cluster]['closest'].append({'text': text, 'distance': distance})
                    cluster_info_temp[cluster]['farthest'].append({'text': text, 'distance': distance})
                
                total_processed += len(texts)
                pbar.update(len(texts))

                del embeddings, texts, labels, distances, batch

                # Periodically update cluster_info
                if total_processed % (128 * batch_size) == 0:
                    for cluster, info in cluster_info_temp.items():
                        cluster_info[cluster]['total_examples'] += info['total_examples']
                        cluster_info[cluster]['sum_distance'] += info['sum_distance']
                        cluster_info[cluster]['closest'].extend(info['closest'])
                        cluster_info[cluster]['farthest'].extend(info['farthest'])
                        cluster_info[cluster]['closest'] = sorted(cluster_info[cluster]['closest'], key=lambda x: x['distance'])[:5]
                        cluster_info[cluster]['farthest'] = sorted(cluster_info[cluster]['farthest'], key=lambda x: x['distance'], reverse=True)[:5]
                    cluster_info_temp.clear()
                    gc.collect()
        finally:
            writer.close()

    # Final update with remaining temp info
    for cluster, info in cluster_info_temp.items():
        cluster_info[cluster]['total_examples'] += info['total_examples']
        cluster_info[cluster]['sum_distance'] += info['sum_distance']
        cluster_info[cluster]['closest'].extend(info['closest'])
        cluster_info[cluster]['farthest'].extend(info['farthest'])

    for cluster in cluster_info:
        if cluster_info[cluster]['total_examples'] > 0:
            cluster_info[cluster]['average_distance'] = cluster_info[cluster]['sum_distance'] / cluster_info[cluster]['total_examples']
        cluster_info[cluster]['closest'] = sorted(cluster_info[cluster]['closest'], key=lambda x: x['distance'])[:5]
        cluster_info[cluster]['farthest'] = sorted(cluster_info[cluster]['farthest'], key=lambda x: x['distance'], reverse=True)[:5]

    # Save cluster information
    # (Centroids got saved already right after fitting)
    with open(cluster_info_path, 'w') as f:
        json.dump(cluster_info, f, indent=2)

    print("Clustering completed.")

if __name__ == "__main__":
    if cluster_centers_path.exists():
        print("Saved cluster centers found. Loading and skipping to finalize clustering...")
        batchified_kmeans.cluster_centers_ = np.load(cluster_centers_path)
        finalize_clustering()
    else:
        print("No cluster centers found. Starting monitor and fit process...")
        monitor_and_fit()

# tmux new -s cluster_pile
# conda activate minipile
# python 03_cluster_pile_embed.py
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t cluster_pile
# tmux list-sessions
# tmux kill-session -t cluster_pile