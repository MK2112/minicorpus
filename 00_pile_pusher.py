import glob
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
from pathlib import Path

# Helper script, push artifacts to Hugging Face

def push_dataset():
    dataset = DatasetDict({'train': load_dataset('parquet', data_files='/vol/tmp/koppelmm/MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_train_shard_*.parquet', split='train'),
                           'validation': load_dataset('parquet', data_files='/vol/tmp/koppelmm/MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_validation_shard_*.parquet', split='train'),
                           'test': load_dataset('parquet', data_files='/vol/tmp/koppelmm/MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_test_shard_*.parquet', split='train')})
    dataset.push_to_hub("Marcus2112/minipile_low-density")

def push_jsonl_dataset():
    dataset = load_dataset("json", data_files=glob.glob(f"/vol/tmp/koppelmm/MiniPile_BatchKMeans_Double/clustering_results/cluster_results_chunk_*.jsonl"))
    dataset.push_to_hub("Marcus2112/pile_dedup_embeddings_clusters_k440")

def push_model():
    api = HfApi()
    local_path = "/vol/tmp/koppelmm/pythia160m_minipile_DensityProportionedHigh_trained"
    repo_id = "Marcus2112/pythia-160m-minipile_low-density"
    print(f"Pushing model to {repo_id}")
    api.create_repo(repo_id, exist_ok=True)
    print("Repo created. Starting to upload ...")
    api.upload_folder(folder_path=local_path, repo_id=repo_id, repo_type="model")

if __name__ == "__main__":
    #push_dataset()
    push_jsonl_dataset()
    #push_model()