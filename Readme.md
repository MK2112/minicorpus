# MiniCorpus

1. Reproducing MiniPile from 'The Pile' from scratch using HuggingFace and Python.
2. Further improving the MiniPile pipeline and creating a more effective version of MiniPile.
3. Applying the optimized pipeline to RefinedWeb.

## How to Use

The Conda environment file is part of this repository. Create the environment using: `conda env create -f minicorpus.yaml`.

**To build your own MiniPile:**

1. Download the Pile Dataset from HuggingFace using `01_get_piles.ipynb`.
2. Embed the Pile Dataset using `03_embed_pile_dedup_turbo.py`. See the bottom of the script for the command to run.
3. Right after `03_embed_pile_dedup_turbo.py` started, run `03_cluster_pile_embed.py` to cluster the embeddings. It was built to run the model fitting in parallel with the embedding script. 
4. After the embedding script finishes, the clustering script will store the centroids and automatically start clustering the embeddings.
5. Once clustering is done, view the `cluster_info_for_inspection.json` file in the `MiniPile_BatchKMeans` folder for manual cluster inspection.
6. Run `03_sort_pile_clusters.py` to sort the clustered embeddings by their cluster into dedicated `jsonl` files.
7. Run either of the `03_distill_pile_embed_*.py` scripts to distill the embeddings into a flavor of MiniPile.
8. (Optional) Run either of the `03_train_160M_*.py` or `03_train_1.4B_*.py` scripts to train a model on the MiniPile. You may need to uncomment the download function for the script to download the untrained model first.
9. (Optional) Use `00_pile_pusher.py` to push any of the artifacts you produced to your HuggingFace account.

**To explore this project itself:**

Extended documentation, benchmark results, and further suggestions are available in the accompanying Jupyter Notebooks.

## Produced Artifacts

### Datasets

- [pile_dedup_embeddings_clusters_k220](https://huggingface.co/datasets/Marcus2112/pile_dedup_embeddings_clusters_k220)
    - contains the idx from The Pile Deduplicated, the associated cluster, and the embedding's distance to the cluster centroid, for each embedding, from a clustering with $k=220$.
- [minipile_reproduction](https://huggingface.co/datasets/Marcus2112/minipile_reproduction)
    - contains the text and pile idx per document of the reproduction of MiniPile.
- [minipile_cluster-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_cluster-proportioned)
    - contains the text and pile idx per document of a MiniPile that was sampled proportionally to the cluster sizes.
- [minipile_loss-sampled](https://huggingface.co/datasets/Marcus2112/minipile_loss-sampled)
    - contains the text and pile idx per document of a MiniPile that was sampled proportionally to the loss of $n=1000$ embeddings per cluster.
- [minipile_density-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_density-proportioned)
    - contains the text and pile idx per document of a MiniPile that was cluster-wise sampled from proportionally to an equally weighted factor of cluster density and cluster size.
    - this produced better results than the reproduction and is smaller in size.
- [minipile_low-density-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_low-density-proportioned)
    - contains the text and pile idx per document of a MiniPile that was cluster-wise sampled from proportionally to a weighted factor of cluster density and cluster size, but biased towards sampling from lower density clusters.
- [pile_dedup_embeddings_clusters_k440](https://huggingface.co/datasets/Marcus2112/pile_dedup_embeddings_clusters_k440)
    - contains the idx from The Pile Deduplicated, the associated cluster, and the embedding's distance to the cluster centroid, for each embedding, from a clustering with $k=440$.
- [minipile_k440](https://huggingface.co/datasets/Marcus2112/minipile_k440)
    - contains the text and pile idx per document of a MiniPile reproduced with a clustering of $k=440$.
- [minipile_k440_density-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_k440_density-proportioned)
    - contains the text and pile idx per document of a MiniPile that was cluster-wise sampled from proportionally to an equally weighted factor of cluster density and cluster size, with a clustering of $k=440$.

### Models

- [pythia-160m-minipile](https://huggingface.co/Marcus2112/pythia-160m-minipile)
- [pythia-160m-minipile_reproduction](https://huggingface.co/Marcus2112/pythia-160m-minipile_reproduction)
- [pythia-1.4b-minipile_reproduction](https://huggingface.co/Marcus2112/pythia-1.4b-minipile_reproduction)
- [pythia-160m-minipile_cluster-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_cluster-proportioned)
- [pythia-160m-minipile_loss-sampled](https://huggingface.co/Marcus2112/pythia-160m-minipile_loss-sampled)
- [pythia-160m-minipile_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_density-proportioned)
- [pythia-160m-minipile_low_density](https://huggingface.co/Marcus2112/pythia-160m-minipile_low-density)
- [pythia-160m-minipile_k440](https://huggingface.co/Marcus2112/pythia-160m-minipile_k440)
- [pythia-160m-minipile_k440_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_k440_density-proportioned)

## Related Work

- [HuggingFace: EleutherAI/the_pile_deduplicated](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated)
- [Datasheet for the Pile (Biderman, et al. 2022)](https://arxiv.org/abs/2201.07311)
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling (Gao, et al. 2020)](https://arxiv.org/abs/2101.00027)
- [The MiniPile Challenge for Data-Efficient Language Models (Kaddour, Jean. 2023)](https://arxiv.org/abs/2304.08442)
- [Pythia: A suite for Analyzing Large Language Models Across Training and Scaling (Biderman, et al. 2023)](https://arxiv.org/abs/2304.01373)
- [Text Embeddings by Weakly-Supervised Contrastive Pre-Training (Wang, et al. 2022)](https://arxiv.org/abs/2212.03533)
- [DeepCore: A Comprehensive Library for Coreset  Selection in Deep Learning (Guo, et al. 2022)](https://arxiv.org/abs/2204.08499)
- [Extracting representative subset from extensive text data for training pre-trained language models (Suzuki, et al. 2023)](https://www.sciencedirect.com/science/article/pii/S0306457322003508)
- [Embedding Compression with Hashing for Efficient Representation Learning in Large-Scale Graph (Yeh, et al. 2022)](https://arxiv.org/pdf/2208.05648)
- [SemDeDup: Data-efficient learning at web-scale through semantic deduplication (Abbas, et al. 2023)](https://arxiv.org/pdf/2303.09540)
- [Hierarchical Sparse Subspace Clustering (HESSC): An Automatic Approach for Hyperspectral Image Analysis (Shahi, et al. 2020)](https://www.mdpi.com/2072-4292/12/15/2421)