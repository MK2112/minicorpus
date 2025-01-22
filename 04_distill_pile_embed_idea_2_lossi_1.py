import os
import gc
import json
import torch
import jsonlines
import numpy as np
import multiprocessing
import pyarrow.parquet as pq
from tqdm import tqdm
from fastparquet import ParquetFile
from pathlib import Path
from typing import Set, Dict, List
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer

## Idea 2:
#   Lossi (Loss-informed Sampling) is a one-shot proxy-based geometric sampling approach that is guided by a loss-based importance heuristic, 
#   deviating from the original distillation process in the following ways:
#
#   - Per cluster: Uniformly sample $n$ (e.g. $1,000$) documents and determine their loss with a small Pythia $70\text{M}$ proxy model
#   - Use the mean loss as a heuristic for the cluster's informativeness and weight the cluster's representation in the final dataset by this value

# This is a lot. To make this resource-effectively applicable, I split this into several scripts.
# This is script 1 of Idea 2:
#   - Per cluster: Uniformly sample $n$ (e.g. $134,318,121\ \text{documents} / (220\ \text{clusters} \times 100)$) documents from each cluster and determine their loss with a small Pythia $70\text{M}$ proxy model
#   - Persist cluster-wise the mean loss as a heuristic for the cluster's informativeness and weight the cluster's sample factor accordingly

@dataclass
class LossiConfig:
    base_dir: Path = Path("/vol/tmp/koppelmm")
    cluster_dir: Path = base_dir / "MiniPile_BatchKMeans/clustering_sorted"
    cluster_info_path: Path = base_dir / "MiniPile_BatchKMeans/clustering_results/cluster_info_for_inspection.json"
    edition: str = "Lossi_1"
    embd_dir: Path = base_dir / "Pile_Deduplicated_Embd"
    output_loss_path: Path = base_dir / f"MiniPile_{edition}/cluster_loss.jsonl"
    proxy_model_path: Path = base_dir / "pythia70m_dedup_pile_half"
    num_clusters: int = 220
    doc_length: int = 512 # Maximum document length for the proxy model (same as the original e5-large would have gotten, so this is deemed representative)
    excluded_clusters: Set[int] = field(default_factory=lambda: {10, 15, 16, 22, 26, 28, 35, 37, 39, 40, 44, 46, 
                                                                 51, 57, 61, 64, 78, 86, 87, 88, 90, 94, 99, 101,
                                                                 102, 103, 111, 114, 152, 155, 163, 166, 167, 181,
                                                                 196, 200, 218, 219}) # field wrapping for multiprocessing compatibility
    pile_documents_count: int = 134_318_121
    train_count: int = 1_000_000
    val_count: int = 10_000
    test_count: int = 500
    documents_per_cluster: int = 1_000
    rng: np.random.Generator = np.random.default_rng(42)

    def __post_init__(self):
        self.output_loss_path.parent.mkdir(parents=True, exist_ok=True)

def process_cluster_texts(args):
    cluster_idx, valid_idxs, sampled_texts = args
    cluster_texts = [txt[0] for txt in sampled_texts if txt[1] in valid_idxs]
    return cluster_idx, cluster_texts

class LossiSampler:
    def __init__(self, config: LossiConfig):
        self.config = config
        self._load_total_cluster_info()
        self._load_proxy_model()
        self._compute_shard_scopes()

    def _load_total_cluster_info(self):
        with open(self.config.cluster_info_path, 'r') as f:
            self.cluster_info = json.load(f)

    def _load_proxy_model(self):
        # Load a small Pythia 70M proxy model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.proxy_model_path, use_fast=True, local_files_only=True, padding=True, truncation=True, max_length=self.config.doc_length)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.config.proxy_model_path, local_files_only=True, low_cpu_mem_usage=True).to(self.device)
        self.model.eval()

    def _calculate_loss(self, texts: List[str], batch_size: int = 4) -> List[float]:
        # Compute the loss for a batch of documents
        losses = []
        len_texts = len(texts)
        for i in tqdm(range(0, len_texts, batch_size)):
            batch_texts = texts[i:i + batch_size] if i + batch_size < len_texts else texts[i:]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=self.config.doc_length, padding=True).to(self.device)
            with torch.no_grad():
                # Process the batch of size 4
                outputs = self.model(**inputs, labels=inputs["input_ids"]) # passing key-value pairs from inputs dict as args to model's forward
                losses.append(outputs.loss.item() / len(batch_texts)) # Normalize loss by batch size
            del inputs, outputs, batch_texts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return losses

    def _read_idxs_for_cluster(self, cluster_idx: int) -> List[int]:
        # Read document indices assoricated with a given cluster
        cluster_file = self.config.cluster_dir / f"cluster_{cluster_idx:03d}.jsonl"
        cluster_idxs = []
        with jsonlines.open(cluster_file) as reader:
            for entry in reader:
                if entry['cluster'] == cluster_idx:
                    cluster_idxs.append(entry['idx'])
        return cluster_idxs

    def _sample_cluster_docs(self, valid_idxs: List[int]) -> List[int]:
        return self.config.rng.choice(valid_idxs, size=min(len(valid_idxs), self.config.documents_per_cluster), replace=False).tolist()

    def _shard_with_idx(self, idx: int) -> int:
        # Find the shard containing a specific index by entry count heuristic
        # Return the shard index as well as the local index within the shard
        for i, _ in enumerate(self.shard_idxs):
            if i + 1 < len(self.shard_idxs) and idx < self.shard_idxs[i + 1]:
                return i
        return len(self.shard_idxs) - 1

    def _read_fast_parquet(self, file_path: str, idxs: List[int], limit: bool = False) -> np.ndarray:
        # This is really fast, really memory-optimized, but bloats the cache; I can't control that.
        parquet = ParquetFile(file_path) # https://github.com/dask/fastparquet/issues/386
        if limit:
            # Limit each string entry from 'text' column to 512 characters
            result = np.vectorize(lambda x: x[:512])(parquet.to_pandas(columns=['text'])['text'].to_numpy()[idxs])
        else:
            result = parquet.to_pandas(columns=['text'])['text'].to_numpy()[idxs]
        del parquet
        return result

    def _get_texts_for_idxs(self, idxs: List[int]) -> List[tuple]:
        # Group indices by shard to minimize I/O
        idxs_by_shard = {}
        for idx in idxs:
            shard_idx = self._shard_with_idx(idx)
            if shard_idx not in idxs_by_shard:
                idxs_by_shard[shard_idx] = []
            idxs_by_shard[shard_idx].append(idx)
        txts_with_idxs = []
        for shard_idx, shard_idxs in tqdm(idxs_by_shard.items(), desc="Reading Texts", unit="doc"):
            shard_idxs.sort() # Sort indices to have some speedup from sequential access
            local_idxs = [idx - self.shard_idxs[shard_idx] for idx in shard_idxs]
            shard_file = Path(self.config.embd_dir) / f"shard_{shard_idx:09d}.parquet"
            shard_texts = self._read_fast_parquet(str(shard_file), idxs=local_idxs, limit=True)
            txts_with_idxs.extend(zip(shard_texts, shard_idxs))
            del local_idxs, shard_texts, shard_file
            gc.collect()
        return txts_with_idxs

    def _compute_shard_scopes(self):
        # Precompute cumulative entries for efficient document lookup
        # Serves to boost lookup performance, and is as hacky as it gets
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

        last_shard_size = self.config.pile_documents_count - cumulative_idxs
        if last_shard_size > 0:
            self.shard_idxs.append(cumulative_idxs)

    def calculate_losses(self):
        if not os.path.exists(self.config.base_dir / f"MiniPile_{self.config.edition}/sampled_texts_dict.json"):
            all_idxs = []
            valid_idxs_cluster = {}
            for cluster_idx in tqdm(range(self.config.num_clusters), desc="Reading Cluster Indices", unit="cluster"):
                if cluster_idx in self.config.excluded_clusters:
                    continue
                valid_idxs = self._read_idxs_for_cluster(cluster_idx)
                valid_idxs_cluster[cluster_idx] = valid_idxs
                all_idxs.extend(self._sample_cluster_docs(valid_idxs))

            sampled_texts = self._get_texts_for_idxs(all_idxs)
            sampled_texts_dict = {}
            
            with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() // 2)) as pool:
                cluster_args = [(cluster_idx, valid_idxs, sampled_texts)
                                for cluster_idx, valid_idxs in valid_idxs_cluster.items()
                                if cluster_idx not in self.config.excluded_clusters]
                # https://superfastpython.com/multiprocessing-pool-imap_unordered/
                # unordered to ... well, keep order (as is, not imposing anything)
                # I had thought I'd need it elsewhere, here its kinda unnecessary; but hey, good habit I guess
                results_iter = tqdm(pool.imap_unordered(process_cluster_texts, cluster_args),
                                    total=len(cluster_args), 
                                    desc="Grouping Sampled Texts", 
                                    unit="cluster")
                sampled_texts_dict = dict(results_iter)
            del sampled_texts, all_idxs, valid_idxs_cluster

            # Persist sampled_texts_dict to disk for if something goes wrong
            with open(self.config.base_dir / f"MiniPile_{self.config.edition}/sampled_texts_dict.json", "w") as f:
                json.dump(sampled_texts_dict, f)
        else:
            print("[~] Loading sampled texts from saved file")
            with open(self.config.base_dir / f"MiniPile_{self.config.edition}/sampled_texts_dict.json", "r") as f:
                sampled_texts_dict = json.load(f)

        results = {}
        for cluster_idx, cluster_texts in tqdm(sampled_texts_dict.items(), desc="Calculating Losses", unit="cluster"):
            loss_mean = np.mean(self._calculate_loss(cluster_texts))
            print(f"Cluster {int(cluster_idx):03d} - Mean Loss: {loss_mean:.5f}")
            results[int(cluster_idx)] = loss_mean
            del cluster_texts
        del sampled_texts_dict
        return results

    def persist_results(self, cluster_losses: Dict[int, float], cluster_proportions: Dict[int, float]) -> None:
        with open(self.config.output_loss_path, "w") as f:
            for cluster_idx, mean_loss in cluster_losses.items():
                f.write(json.dumps({"cluster_idx": cluster_idx, "mean_loss": mean_loss, "proportion": cluster_proportions[cluster_idx]}) + "\n")

    def calculate_proportions(self, cluster_losses: Dict[int, float]) -> Dict[int, float]:
        # Based on the mean losses, calculate the proportion of the dataset each cluster should contribute
        total_count = self.config.train_count + self.config.val_count + self.config.test_count
        total_loss = sum(cluster_losses.values())
        proportions = {}
        for cluster_idx, mean_loss in cluster_losses.items():
            proportions[cluster_idx] = (mean_loss / total_loss) * total_count
        return proportions

if __name__ == "__main__":
    config = LossiConfig()
    sampler = LossiSampler(config)
    cluster_losses = sampler.calculate_losses()
    cluster_proportions = sampler.calculate_proportions(cluster_losses)
    sampler.persist_results(cluster_losses, cluster_proportions)
    print(f"[+] Mean losses saved to {config.output_loss_path}")

# tmux new -s mini_2
# conda activate minicorpus
#
# CUDA_VISIBLE_DEVICES=2 python 04_distill_pile_embed_idea_2_lossi_1.py
#  
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t mini_2
# tmux list-sessions
# tmux kill-session -t mini_2
#
# This took ~4.5 hours on a single A6000