import glob
from pathlib import Path
from huggingface_hub import HfApi
from datasets import load_dataset, DatasetDict

# Helper script, push artifacts to Hugging Face

base_path = Path("/vol/tmp/koppelmm")

def push_dataset():
    dataset = DatasetDict({'train': load_dataset('parquet', data_files=str(base_path / 'MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_train_shard_*.parquet'), split='train'),
                           'validation': load_dataset('parquet', data_files=str(base_path / 'MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_validation_shard_*.parquet'), split='train'),
                           'test': load_dataset('parquet', data_files=str(base_path / 'MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_test_shard_*.parquet'), split='train')})
    dataset.push_to_hub("Marcus2112/minipile_low-density")

def push_jsonl_dataset():
    dataset = load_dataset("json", data_files=glob.glob(str(base_path / "MiniPile_BatchKMeans_Double/clustering_results/cluster_results_chunk_*.jsonl")))
    dataset.push_to_hub("Marcus2112/pile_dedup_embeddings_clusters_k440")

def push_model():
    api = HfApi()
    local_path = str(base_path / "pythia160m_minipile_DensityProportionedHigh_trained")
    repo_id = "Marcus2112/pythia-160m-minipile_low-density"
    api.create_repo(repo_id, exist_ok=True)
    print(f"Pushing model to {repo_id}...")
    api.upload_folder(folder_path=local_path, repo_id=repo_id, repo_type="model")

if __name__ == "__main__":
    #push_dataset()
    push_jsonl_dataset()
    #push_model()