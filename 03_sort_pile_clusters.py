import json
import gc
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Manager, Process, cpu_count

# Run this after "03_cluster_pile_embed.py"
# Run this before "03_distill_minipile.py"
# Sorts the (index-consistent but cluster-wise unsorted) entries from the clustering process to be grouped by clusters.
# Could have been done in "03_cluster_pile_embed.py" already, but this is SoC.

inp_path = Path("/vol/tmp/koppelmm/MiniPile_BatchKMeans/clustering_results")
out_path = Path("/vol/tmp/koppelmm/MiniPile_BatchKMeans/clustering_sorted")
out_path.mkdir(exist_ok=True, parents=True)

k_clusters = 220

def process_file_chunk(f_chunk, queue, pbar_files):
    try:
        local_data = defaultdict(list)
        for f_jsonl in f_chunk:
            with open(f_jsonl, 'r') as jsonl_file:
                for entry in jsonl_file:
                    try:
                        j_entry = json.loads(entry)
                        cluster_id = j_entry['cluster']
                        local_data[cluster_id].append(entry)
                    except json.JSONDecodeError:
                        print(f"Malformed entry {f_jsonl}: {entry}")
            gc.collect()
            pbar_files.update(1)        
        # Queue data for each cluster
        for cluster_id, entries in local_data.items():
            queue.put((cluster_id, entries))
        
    except Exception as e:
        print(f"Error during chunk processing: {e}")
    finally:
        gc.collect()

def write_clusters(queue, f_locks):
    """Write cluster data from queue to files."""
    while True:
        cluster_data = queue.get()
        if cluster_data is None:  # Poison pill for termination
            break

        cluster_id, entries = cluster_data
        cluster_file = out_path / f"cluster_{cluster_id:03d}.jsonl"

        try:
            with f_locks[cluster_id]:
                with open(cluster_file, 'a') as cluster_f:
                    cluster_f.writelines(entries)
        except Exception as e:
            print(f"Error writing to cluster file {cluster_file}: {e}")

def sort_to_clusters(inp_path: Path, out_path: Path, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)
        print(f"[~] Using {num_workers} workers for sorting.")

    jsonl_files = sorted(inp_path.glob("cluster_results_chunk_*.jsonl"))
    jsonl_count = len(jsonl_files)

    chunk_size = (jsonl_count + num_workers - 1) // num_workers
    file_chunks = [jsonl_files[i:i + chunk_size] for i in range(0, jsonl_count, chunk_size)]
    print(f"[~] Split {jsonl_count} JSONL files into {len(file_chunks)} chunks, of size {chunk_size}.")

    with Manager() as manager:
        file_locks = manager.dict({cluster_id: manager.Lock() for cluster_id in range(k_clusters)})
        queue = manager.Queue()
        pbar_files = tqdm(total=jsonl_count, desc="Processing Files", position=0, leave=True)

        # Start writer processes
        writers = []
        for _ in range(num_workers):
            writer = Process(target=write_clusters, args=(queue, file_locks))
            writers.append(writer)
            writer.start()

        # Start reader processes
        processes = []
        for file_chunk in file_chunks:
            p = Process(target=process_file_chunk, args=(file_chunk, queue, pbar_files))
            processes.append(p)
            p.start()

        # Wait for readers to complete
        for p in processes:
            p.join()

        # Signal writers to stop
        for _ in writers:
            queue.put(None)

        # Wait for writers to complete
        for writer in writers:
            writer.join()

        pbar_files.close()

    print("[!] All processes completed.")

if __name__ == "__main__":
    try:
        sort_to_clusters(inp_path, out_path)
        print("[+] Cluster-specific JSONL-Entry Sorting Complete.")
    except Exception as e:
        print(e)

# tmux new -s sort_pile
# conda activate minipile
# python 03_sort_pile_clusters.py
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t sort_pile
# tmux list-sessions
# tmux kill-session -t sort_pile
# 
# Took roughly (hilariously) 1 minute.