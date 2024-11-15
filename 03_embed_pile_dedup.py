import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset,  Dataset
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

base_dir = "/vol/tmp/koppelmm"
base_path = Path(base_dir)

def download_model(down_dir: str, target_folder: str, cache_folder: str, repo_id: str, branch: str = "main") -> None:
    down_dir = Path(down_dir)
    target_dir = down_dir / target_folder
    cache_dir = down_dir / cache_folder

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading {repo_id}/{branch}...")

    while True:
        try:
            snapshot_download(
                repo_id,
                repo_type="model",
                revision=branch,
                cache_dir=str(cache_dir),
                local_dir=str(target_dir)
            )
            break
        except Exception as e:
            print(f"Download attempt failed: {e}")
            continue

# Starting point is the deduplicated The Pile
# Infer embeddings for all documents using E5-Large

# https://huggingface.co/intfloat/e5-large
download_model(down_dir=base_dir, target_folder="e5-large", 
               cache_folder="e5-large_Cache",
               repo_id="intfloat/e5-large") # Chose this because nothing beyond E5-Large was specified

e5_large = SentenceTransformer(str(base_path / "e5-large"), local_files_only=True) # no .from_pretrained() here

# https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated
pile_dedup = load_dataset("parquet",
                          data_files={
                              "train": str(base_path / "Pile_Deduplicated" / "data" / "train-*.parquet"),
                          },
                          cache_dir=str(base_path / "MiniPile_Cache"),
                          split="train",
                          streaming=True)

# Took the example code from the intfloat/e5-large page
embd_dir = base_path / Path("Pile_Deduplicated_Embeddings")
embd_dir.mkdir(exist_ok=True)

batch_size = 1024
shard_size = batch_size ** 2 # shard embed count upper bound

embedding_shard = []
shard_index = 0

def save_shard(embeddings, output_dir, shard_index):
    shard_path = output_dir / f"shard_{shard_index:09d}.parquet"
    dataset = Dataset.from_dict({"embedding": embeddings})
    dataset.to_parquet(str(shard_path))

# Didn't know tqdm could be used like that
for batch_idx, batch in tqdm(enumerate(pile_dedup.iter(batch_size=batch_size))):
    batch_embds = e5_large.encode(batch['text'], show_progress_bar=False) # Set this to False, good for debug but clutters like hell
    embedding_shard.extend(batch_embds)
    
    if len(embedding_shard) >= shard_size:
        save_shard(embedding_shard, embd_dir, shard_index)
        shard_index += 1
        embedding_shard = []

# Append remaining
if embedding_shard != []:
    save_shard(embedding_shard, embd_dir, shard_index)

# tmux new -s embed_pile
# conda activate minipile
# python 03_embed_pile_dedup.py
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t embed_pile
# tmux list-sessions
# tmux kill-session -t embed_pile