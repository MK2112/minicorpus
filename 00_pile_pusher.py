import os
import glob
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from datasets import load_dataset, DatasetDict

# Helper script, push artifacts to Hugging Face

base_path = Path("/vol/tmp/koppelmm")

def push_dataset(repo_id):
    dataset = DatasetDict({'train': load_dataset('parquet', data_files=str(base_path / 'MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_train_shard_*.parquet'), split='train'),
                           'validation': load_dataset('parquet', data_files=str(base_path / 'MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_validation_shard_*.parquet'), split='train'),
                           'test': load_dataset('parquet', data_files=str(base_path / 'MiniPile_DensityProportionedHigh/minipile_DensityProportionedHigh_test_shard_*.parquet'), split='train')})
    dataset.push_to_hub(repo_id)

def push_xl_dataset(repo_id, max_files=None):
    # Dataset is bigger than 300GB, ask datasets@huggingface.co to grant more storage
    # See https://huggingface.co/docs/hub/storage-limits on what to inform about exactly
    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    all_files = sorted(glob.glob(str(base_path / 'RefinedWeb_Embd/shard_*.parquet')))
    files_to_upload = all_files[:max_files] if max_files is not None else all_files
    for file_path in tqdm(files_to_upload):
        file_name = os.path.basename(file_path)
        print(f"Uploading {file_name}...")
        api.upload_file(path_or_fileobj=file_path,
                        path_in_repo=f"{file_name}",
                        repo_id=repo_id,
                        repo_type="dataset")

def push_jsonl_dataset(repo_id):
    dataset = load_dataset("json", data_files=glob.glob(str(base_path / "MiniPile_BatchKMeans_Double/clustering_results/cluster_results_chunk_*.jsonl")))
    dataset.push_to_hub(repo_id)

def push_model(repo_id):
    api = HfApi()
    local_path = str(base_path / "pythia160m_minipile_DensityProportionedHigh_trained")
    api.create_repo(repo_id, exist_ok=True)
    print(f"Pushing model to {repo_id}...")
    api.upload_folder(folder_path=local_path, repo_id=repo_id, repo_type="model")

if __name__ == "__main__":
    #push_dataset("Marcus2112/minipile_low-density")
    #push_jsonl_dataset(repo_id="Marcus2112/pile_dedup_embeddings_clusters_k440")
    #push_model(repo_id="Marcus2112/pythia-160m-minipile_low-density")
    push_xl_dataset(repo_id="Marcus2112/refinedweb-embedded_prototype", max_files=50)