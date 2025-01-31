import gc
import json
import torch
import numpy as np
import jsonlines
import pyarrow as pa
import torch.nn as nn
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from typing import Set, Dict, List
from fastparquet import ParquetFile
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from transformers import AutoModelForCausalLM, AutoTokenizer

## Idea 2:
#   Lossi (Loss-informed Sampling) is a one-shot proxy-based geometric sampling approach that is guided by a loss-based importance heuristic, 
#   deviating from the original distillation process in the following ways:
#
#   Idea 2.1 covers these adaptations:
#   - Per cluster: Uniformly sample $n$ (e.g. $1,000$) documents and determine their loss with a small Pythia $70\text{M}$ proxy model
#   - Use the mean loss as a heuristic for the cluster's informativeness and weight the cluster's representation in the final dataset by this value
#
#   Idea 2.2 covers these adaptations:
#   - The loss-proportional sampling information from Idea 2.1 is used to guide the cluster-wise random sampling process
#   - Per cluster: Randomly sample $1.5\times$ the amount of documents we want to end up with from each non-excluded cluster
#   - Per cluster: Calculate the loss for each sampled document with a small Pythia $70\text{M}$ proxy model which itself was pretrained halfway through (`step72000`) The Pile Deduped.
#   - Per cluster: Sort the documents by their loss and select the top half of the documents with the highest loss for the final dataset
#   - We continue with the dataset assembly as before after that.

# This is a lot. To make this resource-effectively applicable, I split this into several scripts.
# This is script 1 of idea 2.2:
#   - Use the 'cluster_loss.jsonl' to - per cluster - read the count of documents to sample, utilizing loss-proportional sampling information from Idea 2.1
#   - 'cluster_loss.jsonl' entries look like this: {"cluster_idx": 25, "mean_loss": 1.8869157474040985, "proportion": 5637.40015741958}
#   - Per cluster: Randomly sample $1.5\times$ the read-in proportional amount of documents we want to end up with from each non-excluded cluster
#   - Per cluster: Calculate the loss for each sampled document with a small Pythia $70\text{M}$ proxy model which itself was pretrained halfway through (`step72000`) The Pile Deduped.
#   - Per cluster: Sort the documents by their loss and select the top [proporional amount as given by the jsonl] many of the documents with the highest loss for the final dataset
#   - Continue with the dataset assembly as usual after that.

@dataclass
class DistillConfig:
    base_dir: Path = Path("/vol/tmp/koppelmm")
    cluster_dir: Path = base_dir / "MiniPile_BatchKMeans/clustering_sorted"
    cluster_info_path: Path = base_dir / "MiniPile_BatchKMeans/clustering_results/cluster_info_for_inspection.json"
    embd_dir: Path = base_dir / "Pile_Deduplicated_Embd"
    cluster_proportions_path: Path = base_dir / 'MiniPile_Lossi/cluster_loss.jsonl'
    proxy_model_path: Path = base_dir / "pythia70m_dedup_pile_half"
    num_clusters: int = 220 # As per paper
    num_clusters_to_exclude: int = 38 # As per paper
    edition: str = "Lossi_2" # Version of MiniPile, distinguishes file naming + output directory
    excluded_clusters: Set[int] = field(default_factory=lambda: {10, 15, 16, 22, 26, 28, 35, 37, 39, 40, 44, 46, 
                                                                 51, 57, 61, 64, 78, 86, 87, 88, 90, 94, 99, 101,
                                                                 102, 103, 111, 114, 152, 155, 163, 166, 167, 181,
                                                                 196, 200, 218, 219})
    train_count: int = 1_000_000
    val_count: int = 500
    test_count: int = 10_000 # 1M/500/10k Train/Val/Test total (for best comparability to baseline)
    output_shard_size: int = 100_000
    proportion_scaler: float = 1.5  # Oversample each cluster by this factor
    pile_size: int = 134_318_121 # Could've been read from the metadata file, too
    rng: np.random.Generator = np.random.default_rng(42) # reproducibility
    sampling_seq_len: int = 512 # Sequence length for sampling with the proxy model
    batch_size: int = 4 # Batch size during loss calculation

    def __post_init__(self):
        # Just nicer and more distinct to place here
        self.output_dir = self.base_dir / f"MiniPile_{self.edition}"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        # Not used here, but needed later to not hit disk quotas
        self.cache_dir = self.base_dir / f"MiniPile_{self.edition}_Cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

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
        self._load_total_cluster_info()
        self._load_proportion_info()
        self._compute_shard_scopes()
        self._load_proxy_model()
        self.shard_counter: int = 0
        self.writer = MiniCorpusWriter(output_dir=config.output_dir,
                                       edition=config.edition,
                                       output_shard_size=config.output_shard_size)
    
    def _load_proxy_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.proxy_model_path, use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Load the model
        self.proxy_model = AutoModelForCausalLM.from_pretrained(self.config.proxy_model_path, local_files_only=True, low_cpu_mem_usage=True).to(self.device)
        self.proxy_model.eval()
        # Load the loss criterion -> Allows us to retrieve loss per document in batch
        self.criterion = nn.CrossEntropyLoss(reduction='none') # Per-token loss, we can average those per document

    def _load_proportion_info(self):
        # Load cluster proportion information JSONL file, populate class attribute
        with jsonlines.open(self.config.cluster_proportions_path) as reader:
            self.cluster_proportions = {entry['cluster_idx']: int(np.round(entry['proportion'])) for entry in reader}

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
        cluster_idxs = []
        with jsonlines.open(cluster_file) as reader:
            for entry in reader:
                if entry['cluster'] == cluster_idx:
                    cluster_idxs.append(entry['idx'])
        return cluster_idxs
    
    def _sample_cluster_docs(self, valid_idxs: List[int], num_samples: int) -> List[int]:
        return self.config.rng.choice(valid_idxs, size=min(len(valid_idxs), num_samples), replace=False).tolist()
    
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

        # split the shuffled indices proportionally to the original dataset, but dynamically based on this setting's cluster sizes
        # Maintaining the rough train/val/test split from the paper, that is e.g. proportion of 1.000.000 to 1.010.500 for train
        train_count = int(len(idxs_to_shuffle) * 0.9896091)
        val_count = int(len(idxs_to_shuffle) * 0.009896091)
        test_count = len(idxs_to_shuffle) - train_count - val_count

        # Hacky, but ok weight balancing
        while train_count + val_count + test_count < len(idxs_to_shuffle):
            test_count += 1
        while train_count + val_count + test_count > len(idxs_to_shuffle):
            if test_count > 100:
                test_count -= 1
            elif val_count > 5000:
                val_count -= 1
            else:
                train_count -= 1

        return {'train': idxs_to_shuffle[:train_count],
                'validation': idxs_to_shuffle[train_count:(train_count + val_count)],
                'test': idxs_to_shuffle[(train_count + val_count):(train_count + val_count + test_count)]}
    
    def _get_texts_for_idxs(self, idxs: List[int]) -> List[Tuple[int, str]]:
        # Read the texts for a list of indices
        idxs_with_texts = []
        shard_idxs = [self._shard_with_idx(idx) for idx in idxs]
        # Group indices by shard
        idxs_by_shard = {}
        for idx, shard_idx in zip(idxs, shard_idxs):
            if shard_idx not in idxs_by_shard:
                idxs_by_shard[shard_idx] = []
            idxs_by_shard[shard_idx].append(idx)
        # Read the texts for each shard
        for shard_idx, idxs in idxs_by_shard.items():
            shard_file = Path(self.config.embd_dir) / f"shard_{shard_idx:09d}.parquet"
            local_idxs = [idx - self.shard_idxs[shard_idx] for idx in idxs]
            idxs_with_texts.extend([(idx, text) for idx, text in zip(idxs, self._read_fast_parquet(str(shard_file), local_idxs, limit=True))])
        return idxs_with_texts

    def _calculate_losses(self, idxs_with_texts: List[Tuple[int, str]]) -> List[Tuple[int, float]]:
        idxs_with_losses = []
        len_texts_with_idxs = len(idxs_with_texts)
        for i in tqdm(range(0, len_texts_with_idxs, self.config.batch_size), desc="Calculating Losses", unit="batch"):
            batch = idxs_with_texts[i:i+self.config.batch_size] if i + self.config.batch_size < len_texts_with_idxs else idxs_with_texts[i:]
            batch_texts, batch_idxs = zip(*batch) # [(idx, text), ...] -> [text, ...], [idx, ...]
            inputs = self.tokenizer(list(batch_texts), return_tensors='pt', truncation=True, max_length=self.config.sampling_seq_len, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.proxy_model(**inputs, labels=inputs['input_ids'])
                # Calculating the loss for each document, averaged over its tokens
                token_lvl_losses = self.criterion(outputs.logits.view(-1, outputs.logits.shape[-1]), inputs['input_ids'].view(-1))
                # Average loss per document
                mean_losses = token_lvl_losses.view(len(batch_texts), -1).mean(dim=-1).cpu().numpy()
                idxs_with_losses.extend(zip(batch_idxs, mean_losses))
            del inputs, outputs, token_lvl_losses, batch_texts, batch_idxs, mean_losses
            gc.collect() # I'm paranoid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return idxs_with_losses

    def _process_cluster(self, cluster: int) -> Set[int]:
        if cluster not in self.config.excluded_clusters:
            # Get document indices associcated with cluster
            valid_idxs = self._read_idxs_for_cluster(cluster)
            # Oversampling the pre-computed cluster proportion, while ensuring we don't exceed available indices
            oversampled_idxs = self._sample_cluster_docs(valid_idxs, min(self.cluster_proportions[cluster] * self.config.proportion_scaler, len(valid_idxs)))
            sampling_bound = min(self.cluster_proportions[cluster], len(valid_idxs))
            del valid_idxs
            # Assemble the texts for the oversampled indices
            oversampled_idxs_with_texts = self._get_texts_for_idxs(oversampled_idxs)
            del oversampled_idxs
            # Calculate the loss for each document
            oversampled_idxs_with_losses = self._calculate_losses(oversampled_idxs_with_texts)
            del oversampled_idxs_with_texts
            # Sort the documents by loss, descending, and select the top many as given by the cluster proportion
            sorted_idxs_with_losses = sorted(oversampled_idxs_with_losses, key=lambda x: x[1], reverse=True)[:sampling_bound]
            del oversampled_idxs_with_losses, sampling_bound
            gc.collect()
            return set([idx for idx, _ in sorted_idxs_with_losses])
        else:
            return set()

    def build_minicorpus(self):
        results = [self._process_cluster(cluster) for cluster in range(self.config.num_clusters)]
        # Combine results from all workers
        idxs_to_extract = sorted(set().union(*results))
        del results
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
        assert sum(len(idxs) for idxs in idxs_by_shard.values()) == len(split_idxs_to_extract)
        
        # These are the Parquet files that contain the documents
        parquet_files = sorted(Path(self.config.embd_dir).glob("shard_*.parquet"))
        for shard_idx, idxs in tqdm(idxs_by_shard.items(), desc=f"Processing {split_name} Shards", unit="shard"):
            # Read the indices accumulated for each shard and persist them to a new Parquet file
            self._process_shard(shard_idx, idxs, parquet_files[shard_idx], split_name)
        del idxs_by_shard
        gc.collect()
        # Pack up your stuff, we're done with the split
        self.writer.finalize()

    def _read_fast_parquet(self, file_path: str, idxs: List[int], limit: bool = False) -> np.ndarray:
        # As fast and as memory-efficient a Parquet reader as I could come up with.
        # Stackoverflow throws the towel on this one, but here, memory use except cache is actually quite low, yet cache is a killer. 
        # That's on pyarrow/fastparquet.
        parquet = ParquetFile(file_path)
        if limit:
            # Limit each string entry from 'text' column to sampling_seq_len characters
            result = np.vectorize(lambda x: x[:self.config.sampling_seq_len])(parquet.to_pandas(columns=['text'])['text'].to_numpy()[idxs])
        else:
            # Running with pandas instead of numpy is slower here by ~10%
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
        # selected the columns and filter down to the entries I actually need.
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

# tmux new -s mini_22_1
# conda activate minipile
#
# CUDA_VISIBLE_DEVICES=2 python 03_distill_pile_embed_idea_2.2_lossi_1.py
#
# Detach from tmux session: Ctrl-b followed by d
# Reattach to tmux session: tmux attach -t mini_22_1
# tmux list-sessions
# tmux kill-session -t mini_22_1