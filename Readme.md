# MiniCorpus

MiniCorpus reproduces and enhances [MiniPile (Kaddour, Jean. 2023)](https://arxiv.org/abs/2304.08442), a distilled subset of [The Pile Deduplicated](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated). MiniPile enables efficient language model training using two orders of magnitude less data while maintaining competitive performance compared to models trained on the full deduplicated Pile.

MiniCorpus covers the following steps:

1. Reproducing MiniPile from 'The Pile Deduplicated' from scratch, using HuggingFace and PyTorch.
2. Further improving the MiniPile pipeline and creating a more effective version of MiniPile.
3. Preparing the optimized pipeline for general applicability with the example of [RefinedWeb (Penedo, et al. 2023)](https://arxiv.org/abs/2306.01116).

## Project Setup

Create the project's conda environment using: `conda env create -f minicorpus.yaml`.

### Quick Guide: Build your own MiniPile

1. Download [The Pile Deduplicated](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated) from HuggingFace, e.g. by using `01_get_piles.ipynb`.
2. Embed the deduplicated Pile using `03_embed_pile_dedup_turbo.py`. See the bottom of the script on how to run it.
3. Right when `03_embed_pile_dedup_turbo.py` starts processing, run `03_cluster_pile_embed.py` to cluster the embeddings. The clustering script was built to run centroid discovery in parallel to the embedding script producing new embeddings.
4. After the embedding script finishes, `03_cluster_pile_embed.py` will store the centroids and automatically start clustering the embeddings.
5. Once clustering concluded, you may inspect the generated `cluster_info_for_inspection.json` in the `MiniPile_BatchKMeans` folder for manual cluster inspection and exclusion.
6. Run `03_sort_pile_clusters.py` to sort the clustered embeddings by their cluster into dedicated `jsonl` files.
7. Run either `03_distill_pile_embed.py` or either of the `04_distill_pile_embed_*.py` scripts to sample a flavor of MiniPile from the embeddings.
8. (Optional) Run `03_train_160M_*.py` or `03_train_1.4B_*.py` or either of the `04_train_160M_*.py` or `04_train_1.4B_*.py` to train the model on your MiniPile flavor. You may need to uncomment the download function inside the training script to have it download the untrained model first.
9. (Optional) Use `00_pile_pusher.py` to push any of the artifacts you produced to your HuggingFace account.

## Reproducing MiniPile

The reproduction process is split across three chapters. Files belonging to these chapters are prefixed with `01_`, `02_`, and `03_` respectively.<br>
Jupyter Notebooks are added for each chapter to guide you through the process.

- Chapter `01` is concerned with downloading The Pile Deduplicated and the original MiniPile (for comparison). Be sure to have enough disk space available.
    - The guide is available in the Jupyter Notebook `01_get_piles.ipynb`.
- Chapter `02` is concerned with training a [Pythia](https://arxiv.org/abs/2304.01373) [160M](https://huggingface.co/EleutherAI/pythia-160m) model on the original MiniPile and benchmarking it with the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). 
    - The guide is available in the Jupyter Notebook `02_eval_160M.ipynb`.
- Chapter `03` is concerned with reproducing MiniPile from scratch. This includes embedding The Pile Deduplicated, clustering the embeddings, and sampling a MiniPile from the clusters in accordance with the [original paper](https://arxiv.org/abs/2304.08442).

## Improving the MiniPile Pipeline, Practically

The MiniPile pipeline can be improved by sampling a datasubset that is ideally even smaller than MiniPile and yet more representative of the original Pile Deduplicated.<br>
Ultimately resulting in success, several attempts were undertaken to improve the MiniPile pipeline.<br>
All ideas are documented in the Jupyter Notebook `04_improve_minipile.ipynb`:

1. Cluster-Proportionate Sampling (`04_distill_pile_embed_idea_1_proportionate.py`)
2. Hybrid Loss-Based Sampling (`04_distill_pile_embed_idea_2_lossi_1.py` and `04_distill_pile_embed_idea_2_lossi_2.py`)
3. Size-Density-Proportionate Sampling (`04_distill_pile_embed_idea_3_density.py`)
3.1. Low-Density-Proportionate Sampling (`04_distill_pile_embed_idea_3.1_density_low.py`)
4. Higher-Resolution Clustering (`04_cluster_pile_embed_idea_4_double.py`, `04_distill_pile_embed_idea_4_k440.py`)
5. Higher-Resolution Clustering and Size-Density-Proportionate Sampling (`04_distill_pile_embed_idea_5_k440_density.py`)
6. Inter-Intra-Cluster Sampling with High Clustering Resolution (`04_distill_pile_embed_idea_6_inter.py`)
6.1. Inter-Intra-Cluster Sampling with Inter-Cluster Diversity Weighting Increased (`04_distill_pile_embed_idea_6.1_inter_high.py`)
7. Down-Sized Size-Density-Proportionate Sampling (`04_distill_pile_embed_idea_7_density-tiny.py` and `04_distill_pile_embed_idea_7_density-nano.py`)

Benchmark results for each attempt are available in the notebook and in the [benchmarks](./benchmarks/) folder.<br>
We deem the `Size-Density-Proportionate Sampling` (Idea 3) as the most impactful, as it is the most representative of the original Pile Deduplicated while being smaller in example count than MiniPile. Strongest improvements were observed on the Lambada (Std) benchmark with an improvement of over 50% in perplexity. This approach was further used in (Idea 7) to reduce the distilled, density-sampled dataset size to 90% of the dataset created in (Idea 3).<br>
Even the tiny downsized version of the dataset is at least equal to MiniPile on all benchmarks except for MMLU, and improves its performance on the Lambada (Std) benchmark to 53% better perplexity.

## Improving the MiniPile Pipeline, Theoretically

All of the above improvements and modifications aim to be specifically applicable within resource-constrained (e.g. academic) environments.<br>
At a minimum, you only need disk space for The Pile Deduplicated, its embedded version (I create a copy of Pile Deduplicated because I want to ensure index consistency), the clustering results and the MiniPile you want to sample from it. You will need a GPU for the embedding and clustering steps, but the sampling can be done on a CPU-only machine.

Imposing this constraint for accessibility naturally limits the reachable improvement space for the MiniPile pipeline.<br>
The `04_improve_minipile.ipynb` notebook contains a theoretical section that discusses more fundamental improvements that could be made if the resource constraint was lifted.<br>
These theoretical improvements for assembly are:

- "Sparse Sampling of Embeddings with Similarity Hashing", related to:
    - https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/
    - https://arxiv.org/pdf/2208.05648
    - https://arxiv.org/pdf/1408.2927
    - https://mediatum.ub.tum.de/doc/1655492/ptz317jfxiatpjlwxrpcuimu1.2022-11-01_einreichung_mediatum_titelblatt_neu.pdf

- "Double-Proxied Cross-Entropy-based Sampling", related to:
    - https://www.sciencedirect.com/science/article/pii/S0306457322003508

- "Semantic Deduplication as Post-Processing for the distilled dataset", related to:
    - https://arxiv.org/pdf/2303.09540
    - deemed necessary as the clustering process still indicated presence of duplicates in Pile Deduplicated, this could be mitigated on the distilled dataset level.

## Preparing for general Applicability

The final step of the MiniCorpus project is to prepare the optimized pipeline for general applicability with the example of RefinedWeb.<br>
The RefinedWeb dataset is a subset of the CommonCrawl dataset that was deduplicated.<br>
However, RefinedWeb is not a sum of diverse, smaller datasets like The Pile Deduplicated, but a single, large dataset.<br>
Therefore, we have to find a way lift the need for $k$-means clustering and instead use a more general approach to sample a MiniRefinedWeb.
The mending and adapting of the pipeline for RefinedWeb is documented in the Jupyter Notebook `05_refinedweb_pipeline.ipynb`.

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
- [minipile_low-density-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_low-density-proportioned)
    - contains the text and pile idx per document of a MiniPile that was cluster-wise sampled from proportionally to a weighted factor of cluster density and cluster size, but biased towards sampling from lower density clusters.
- [pile_dedup_embeddings_clusters_k440](https://huggingface.co/datasets/Marcus2112/pile_dedup_embeddings_clusters_k440)
    - contains the idx from The Pile Deduplicated, the associated cluster, and the embedding's distance to the cluster centroid, for each embedding, from a clustering with $k=440$.
- [minipile_k440](https://huggingface.co/datasets/Marcus2112/minipile_k440)
    - contains the text and pile idx per document of a MiniPile reproduced with a clustering of $k=440$.
- [minipile_k440_density-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_k440_density-proportioned)
    - contains the text and pile idx per document of a MiniPile that was cluster-wise sampled from proportionally to an equally weighted factor of cluster density and cluster size, with a clustering of $k=440$.
- [minipile_k440_inter-density-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_k440_inter-density-proportioned)
    - contains the text and pile idx per document of a MiniPile that was cluster-wise sampled from proportionally to an equally weighted set of factors of cluster density, cluster size and inter-cluster diversity, with a clustering of $k=440$.
- [minipile_k440_high-inter_density](https://huggingface.co/datasets/Marcus2112/minipile_k440_high-inter_density)
    - contains the text and pile idx per document of a MiniPile that was cluster-wise sampled from proportionally to an unequally weighted set of factors of cluster density, cluster size and a higher weighted inter-cluster diversity, with a clustering of $k=440$.
- [minipile_density-proportioned_tiny](https://huggingface.co/datasets/Marcus2112/minipile_density-proportioned_tiny)
    - contains the text and pile idx per document of a MiniPile that was cluster-wise sampled from proportionally to an equally weighted factor of cluster density and cluster size, reduced in total example count to 90% of the above density-proportioned MiniPile.
- [minipile_density-proportioned_nano](https://huggingface.co/datasets/Marcus2112/minipile_density-proportioned_nano)
    - retaining only 75% of the original MiniPile example count.

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
- [pythia-160m-minipile_k440_inter-density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_k440_inter-density-proportioned)
- [pythia-160m-minipile_k440_high-inter_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_k440_high-inter_density-proportioned)
- [pythia-160m-minipile_tiny_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_tiny_density-proportioned)

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