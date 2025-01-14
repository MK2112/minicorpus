# MiniCorpus

MiniCorpus reproduces and enhances [MiniPile (Kaddour, Jean. 2023)](https://arxiv.org/abs/2304.08442), a distilled subset of [The Pile Deduplicated](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated). MiniPile enables efficient LLM training using two orders of magnitude less data while maintaining competitive performance compared to models trained on the full deduplicated Pile.

MiniCorpus covers the following steps:
1. Reproducing MiniPile from 'The Pile Deduplicated' from scratch, using HuggingFace and PyTorch.
2. Further improving the MiniPile pipeline and creating more effective versions of MiniPile.
3. Preparing the optimized pipeline for general applicability with the theoretical example of [RefinedWeb (Penedo, et al. 2023)](https://arxiv.org/abs/2306.01116).

## Project Setup

Create the project's conda environment using: `conda env create -f minicorpus.yaml`.

### Quick Guide: Build your own MiniPile

Every script in this respository has the instructions on how to run it at the bottom.

1. Download [The Pile Deduplicated](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated) from HuggingFace, e.g. by using `01_get_piles.ipynb`.
2. Embed the deduplicated Pile using `03_embed_pile_dedup_turbo.py`.
3. Right when `03_embed_pile_dedup_turbo.py` starts processing, run `03_cluster_pile_embed.py` to cluster the embeddings. The clustering script was built to run fitting of k-Means in parallel with the embedding script producing new embeddings.
4. After the embedding script finishes, `03_cluster_pile_embed.py` will store the centroids and automatically start clustering the embeddings.
5. Once clustering concluded, you may inspect the generated `cluster_info_for_inspection.json` in the `MiniPile_BatchKMeans` folder for manual cluster inspection and exclusion.
6. Run `03_sort_pile_clusters.py` to have the clustered embeddings sorted by their assigned cluster into dedicated `jsonl` files.
7. Run `03_distill_pile_embed.py` or either of the `04_distill_pile_embed_*.py` scripts to sample a flavor of MiniPile from the embedded Pile.
8. (Optional) Run `03_train_160M_*.py` or `03_train_1.4B_*.py` or either of the `04_train_160M_*.py` or `04_train_1.4B_*.py` to train a model on your chosen MiniPile flavor. You may need to uncomment the download function inside the training script to have it download the untrained base model first.
9. (Optional) Use `00_pile_pusher.py` to push any of the artifacts you produced to your HuggingFace account.

## Reproducing MiniPile

The reproduction process is split across three chapters.<br>
Files belonging to these chapters are prefixed with `01_`, `02_`, and `03_` respectively.<br>
Jupyter Notebooks are added for each chapter for documentation and to guide you through the process.

- Chapter `01` is concerned with downloading The Pile Deduplicated and the original MiniPile (for comparison). Be sure to have enough disk space available.
    - The guide is available in the Jupyter Notebook `01_get_piles.ipynb`.
- Chapter `02` is concerned with training a [Pythia](https://arxiv.org/abs/2304.01373) [160M](https://huggingface.co/EleutherAI/pythia-160m) model on the original MiniPile and benchmarking it with the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). 
    - The guide is available in the Jupyter Notebook `02_eval_160M.ipynb`.
- Chapter `03` is concerned with reproducing MiniPile from scratch. This includes embedding The Pile Deduplicated, clustering the embeddings, and sampling a MiniPile from the clusters in accordance with the [original paper](https://arxiv.org/abs/2304.08442).

## Improving the MiniPile Pipeline, Practically

The MiniPile pipeline can be improved by sampling a data subset that is ideally even smaller than MiniPile and yet more representative of the original Pile Deduplicated.<br>
Ultimately resulting in success, several attempts were undertaken to improve the MiniPile pipeline for these objectives.<br>
All ideas are documented in the fourth chapter's Jupyter Notebook `04_improve_minipile.ipynb`:

1. Cluster-Proportionate Sampling (`04_distill_pile_embed_idea_1_proportionate.py`)
2. Hybrid Loss-Based Sampling (`04_distill_pile_embed_idea_2_lossi_1.py` and `04_distill_pile_embed_idea_2_lossi_2.py`)
3. Size-Density-Proportionate Sampling (`04_distill_pile_embed_idea_3_density.py`)<br>
3.1. Low-Density-Proportionate Sampling (`04_distill_pile_embed_idea_3.1_density_low.py`)
4. Higher-Resolution Clustering (`04_cluster_pile_embed_idea_4_double.py`, `04_distill_pile_embed_idea_4_k440.py`)
5. Higher-Resolution Clustering and Size-Density-Proportionate Sampling (`04_distill_pile_embed_idea_5_k440_density.py`)
6. Inter-Intra-Cluster Sampling with High Clustering Resolution (`04_distill_pile_embed_idea_6_inter.py`)<br>
6.1. Inter-Intra-Cluster Sampling with Inter-Cluster Diversity Weighting Increased (`04_distill_pile_embed_idea_6.1_inter_high.py`)
7. Down-Sized Size-Density-Proportionate Sampling (`04_distill_pile_embed_idea_7_density-tiny.py`, `04_distill_pile_embed_idea_7_density-nano.py` and `04_distill_pile_embed_idea_7_density-pico.py`)

Benchmark results for each attempt are available in the fourth chapter's Jupyter Notebook and in the [benchmarks](./benchmarks/) folder.<br>
We deem the `Size-Density-Proportionate Sampling` (Idea 3) as the most impactful, as it is the most representative of the original Pile Deduplicated while being smaller in example count than MiniPile. Strongest improvements were observed on the Lambada (Std) benchmark with an improvement of over 50% in perplexity. This approach was further used in (Ideas 7, 8, 9) to reduce the distilled, density-sampled dataset size to 90% (Idea 7) of the dataset created in (Idea 3), and $75\%$ (Idea 8) as well as $25\%$ (Idea 9) of the original MiniPile, respectively.<br>
Even the downsized versions of the dataset are at least equal to MiniPile on all but the MMLU benchmarks.<br>
We even see the reduced (Idea 8) improve its performance on the Lambada (Standard) benchmark to 53% better perplexity than MiniPile.

## Benchmark Results

![](./img/160m_benchmark_results.png)

![](./img/160m_ablation_benchmark_results.png)

![](./img/1.4b_benchmark_results.png)


## Interpretation on practical improvements

With this study project, we successfully replicated the MiniPile pipeline, produced a reproduction dataset and attempted several ideas for improvement of the distillation pipeline, which we then compared primarily by training on the 160M Pythia model architecture.

### Reproduction Challenges and Insights

The reproduction of MiniPile was successful. However, our reproduction had to make multiple assumptions and compromises:
1. Embedding with E5-Large was replaced with E4-Base-4k, which was smaller and faster, but reported to perform worse than E5-Large representation-wise. We addressed this by raising the context size from 512 tokens to 1024 tokens.
2. Step count for learning rate scheduling was not reported, but was scaled to 1024 based on the original dataset's size, intending a most immediate comparability.
3. Cluster exclusion was done manually, as per paper, and while we found and excluded the exact same amount of clusters and the same clusters listed as examples, differences in cluster selection might have occured.

While it wasn't intended, the reproduction dataset showed improvements regarding the perplexity scores on the Lambada (Standard + OpenAI) benchmarks, HellaSwag and ARC-Challenge of up to 2.23%. We consider this to be within the margin of error and therefore not significant.

### 160M Benchmark Insights

Excluding the model versions pre-trained on The Pile, the project produced a Pythia 160M and a Pythia 1.4B trained on the original MiniPile as baselines.<br>
Additionally the project produced 6 160M parameter models exploring ideas for improvement and another 7 160M ablation models, concerning scaling effects of distillation parameters, the distillation dataset size and the training step count. An additional 3 1.4B parameter models were trained with the reproduction MiniPile, the most promising distillation candidate and the lowest example count distillation dataset. A total of 14 datasets has been released in context with the project and released on [HuggingFace](https://huggingface.co/Marcus2112), along with all trained models. (Individual links to models and datasets are listed further below.)

For the 160M model benchmarks, we see that all of the 'improved' MiniPile model variants maintain similar MMLU performance (0.230), undercutting the original MiniPile (0.270) slightly. WinoGrande shows high consistency across MiniPile model variants, too. Compared to the Pile-trained model, every MiniPile model variant indicates catastrophic failure on both versions of Lambada, with perplexity scores in the millions. However, Blimp scores show only moderate degradation from the Pile-trained baseline, indicating a high retention of reasoning capabilities.

The 160M MiniPile base version shows highest MMLU (0.270) and competitive ARC-Challenge scores (0.213), while 160M Density shows best WinoGrande (0.520) and solid Blimp (0.550) scores, and at the same time being approx. 6% smaller in training example count compared to MiniPile.

Of the attempted improvements, the Low Density version shows overall lowest scores, beating the reproduction only slightly in the Blimp benchmark.
This implies that it is important to sufficiently capture dense, similar regions of data instead of overly prefering sparse, unique example regions.

When only seeing these 160M results, one could interpret that:
- Dataset size scales with core capabilities, but only to a disproportionately lower degree
- Quality and representation capability of examples matters more than quantity
- Dense regions of data contain learning signal, while sparse regions contain diversity, which we also have to make sure to consider and include

Notably, Density Tiny, Nano and Pico versions show performance degradation, but this turns out surprisingly minimal.

Considering only the 160M results, one could think that this would be due to:
- The deduplicated Pile potentially still containing duplicates, granting the maintaining of signal strength on smaller subsets
- Density-based sampling identifying truly fundamental examples that encode key patterns
- Model capacity as a limiting factor rather than data, and therefore larger datasets or longer training runs may lead the 160M architecture to overfit

### 1.4B Results and Revision of Hypotheses

Pile Dedup 1.4B shows major improvements over Pile Dedup 160M across all metrics. However, like the original MiniPile, none of the new sampling approaches successfully preserve the qualities that enable scaling with model size. Moreover, none of the MiniPile variants of 1.4B show notable improvements to their 160M counterparts, but instead indicate slight degradation. While training on the Pile Dedup has scores effectively scale with model size (e.g., HellaSwag: 0.29→0.418), all MiniPile variants miss out on this effect and fail to leverage increased model capacity. HellaSwag stays ~0.26, Density-based sampling had seen 0.520 on 160M WinoGrande, but this got even reduced to 0.504 on 1.4B. These results strongly suggest that optimal training of large models requires substantially more, diverse data than any of the sampling methods preserve, particularly for capturing the patterns that larger models can learn to leverage.

Ignoring the results for 1.4B Pythia for The Pile Deduplicated for a moment, we also see another effect: The performance differences between the different MiniPile versions diminish, with the density-based sampling approach not being a clear improvement at all anymore, but instead being marginally better only in HellaSwag and both Lambada perplexity scores.

Therefore, other than the 160M benchmark results alone would let suggest, there exists a more distinct relationship between dataset size and model capacity when training parameter counts are scaled up. From that perspective, the MiniPile variants seem to be creating a "distilled" or "compressed" version of the Pile's knowledge that is particularly well-suited for smaller model capacities. 

With the 1.4B results, we have to revise the 160M result interpretations and have to hypothesize that:
- The strong 160M MiniPile performances likely indicate a model-side capacity bound, rather than solely reflecting dataset efficiency
- The small performance gaps at 160M scale (compared to The Pile Deduplicated) could be misleading about dataset quality
- The Tiny/Nano/Pico versions weren't necessarily "sufficient", but the 160M architecture was plainly too small to indicate a difference; The reduced size may just have made the individual examples more digestible to the small model.

At larger scale, none of the sampling methods come close to full dataset performance. The 1.4B results suggest that proxy-based geometric sampling as performed by MiniPile may be insufficient for building truly scalable, preserving, yet smaller datasets.

### Step Count Ablation Studies

Two ablation studies on 160M Density and 160M Density Pico were conducted with an increased step count from the original 1.5 to now 2 epochs each.
If the compressed datasets truly were to contain more concentrated, representative knowledge, then:

1. If the model hasn't reached its architectural capacity limit:
    - Additional training epochs would allow it to better absorb the concentrated information
    - We would see improved or at worst equal benchmark results (indicating a plateau)
2. If we have the model already at its architectural capacity limit:
    - Additional training would lead to overfitting
    - We would see deteriorating benchmark results

The test revealed nuances: For 160M Density Pico 2 Epochs, we observe stable performance with most score changes below 1%. Accuracy and perplexity scores alike show signs both of improvement and degradation. Still, this stability across metrics suggests the 160M model reached an optimal point in its learning capacity at just 488 steps, instead of the original 1024 used for MiniPile-scale training.<br>
The picture evolves drastically when we examine 160M Density 2 Epochs at 1366 steps. Here, we observe a clear pattern of overfitting on reasoning tasks, yet intriguingly, the model shows substantial improvements in perplexity scores too.

From that, we derive:
1. More training on the distilled MiniPile Density dataset leads to selective overfitting - while the model's language modeling capabilities (measured by perplexity) continue to improve, its performance on reasoning tasks deteriorates.
2. The smaller MiniPile Density Pico dataset appears to provide an actually more concentrated learning signal, reaching optimal performance earlier and maintaining stability with extended training and even improving on perplexity scores compared to standard training on MiniPile Density.
3. Dataset size reduction, at least for The Pile Deduplicated as base, may actually serve as a regularizer, helping prevent overfitting on reasoning tasks while maintaining core language modeling capabilities.

At 160M parameters, the models can effectively learn from this concentrated form of the data without overfitting, possibly because, as we observed, the reduced dataset matches better with what a 160M model can absorb and utilize. When we scale up to 1.4B parameters, this same "compression" becomes a critical limitation - the larger models have the capacity to learn more complex patterns and relationships that were filtered out by the sampling processes.

### Conclusion

With a successful replication and several different improvement approaches explored, a best possible improvement was found with the [minipile_density-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_density-proportioned). This dataset is smaller than MiniPile, but shows competitive performance on all benchmarks.

The behaviors of the different datasets across the 160M and 1.4B architectures suggest that the relationship between factors like dataset size, training duration, and model capacity is more complex than initially thought. In this regime, the Density sampling approach appears effective when combined with aggressive dataset size reduction.

Summarizing the key findings:
> A dataset distillation's performance scales inversely with the size of the model that it will be applied to.<br>
> Reducing the distillate size can serve as a regularizer against overfitting, as epoch count increases.<br>
> Within the Pythia 160M model family, the optimal dataset size may actually be smaller than previously assumed with MiniPile, as signs of overfitting could be observed, and saturation was witnessed with earlier steps.<br>
> The suggested best sampling approach, i.e. weight-density-based sampling, is a promising candidate for further exploration.

## Improving the MiniPile Pipeline, Theoretically

All of the above improvements and modifications aim to be specifically applicable within resource-constrained (e.g. academic) environments.<br>
At a minimum, you only need disk space for The Pile Deduplicated, its embedded version (I create a copy of Pile Deduplicated because I want to ensure index consistency), the clustering results and the MiniPile you want to sample from it. You will need a GPU for the embedding and clustering steps, but the sampling can be done on a CPU-only machine.

Imposing this constraint for accessibility naturally limits the reachable improvement space for the MiniPile pipeline.<br>
The `04_improve_minipile.ipynb` notebook contains a theoretical section that discusses more fundamental improvements that could be made if the resource constraint was lifted.<br>
These theoretical improvements for assembly are:

- "Sparse Sampling of Embeddings with Similarity Hashing", related to:
    - [Locality Sensitive Hashing (LSH): The Illustrated Guide (pinecone.io)](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)
    - [Embedding Compression with Hashing for Efficient Representation Learning in Large-Scale Graph (Yeh, et al. 2022)](https://arxiv.org/pdf/2208.05648)
    - [Hashing for Similarity Search: A Survey (Wang, et al. 2014)](https://arxiv.org/pdf/1408.2927)
    - [Efficient Measures for Processing and Exploring Large-Scale Data Domains (Reinbold, Christian. 2022)](https://mediatum.ub.tum.de/doc/1655492/ptz317jfxiatpjlwxrpcuimu1.2022-11-01_einreichung_mediatum_titelblatt_neu.pdf)

- "Double-Proxied Cross-Entropy-based Sampling", related to:
    - [Extracting representative subset from extensive text data for training pre-trained language models (Suzuki, et al. 2023)](https://www.sciencedirect.com/science/article/pii/S0306457322003508)

- "Semantic Deduplication as Post-Processing for the distilled dataset", related to:
    - [SemDeDup: Data-efficient learning at web-scale through semantic deduplication (Abbas, et al. 2023)](https://arxiv.org/pdf/2303.09540)

## Preparing for general applicability

The final step of the MiniCorpus project is to prepare the optimized pipeline for general applicability with the example of RefinedWeb.<br>
The RefinedWeb dataset is a subset of the CommonCrawl dataset that was deduplicated.<br>
However, RefinedWeb is not a sum of diverse, smaller datasets like The Pile Deduplicated, but a single, large dataset.<br>
Therefore, we have to find a way lift the need for $k$-means clustering and instead use a more general approach to sample a MiniRefinedWeb.
The mending and adapting of the pipeline for RefinedWeb is documented in the Jupyter Notebook `05_refinedweb_pipeline.ipynb`.

## Produced Artifacts

- Go with [minipile_density-proportioned](https://huggingface.co/datasets/Marcus2112/minipile_density-proportioned) if you want to attain the highest benchmark results with lower than 1M examples.
- Go with [minipile_density-proportioned_nano](https://huggingface.co/datasets/Marcus2112/minipile_density-proportioned_nano) if you want to attain largely similar benchmark results to the original MiniPile with less than 750k examples.

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
    - retaining 75% of the original MiniPile example count, sampled with size-density-proportionate sampling at $\omega = 0.5$.
- [minipile_density-proportioned_pico](https://huggingface.co/datasets/Marcus2112/minipile_density-proportioned_pico)
    - retaining 25% of the original MiniPile example count, sampled with size-density-proportionate sampling at $\omega = 0.5$.

### Models

- [pythia-160m-minipile](https://huggingface.co/Marcus2112/pythia-160m-minipile)
- [pythia-160m-minipile_reproduction](https://huggingface.co/Marcus2112/pythia-160m-minipile_reproduction)
- [pythia-160m-minipile_cluster-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_cluster-proportioned)
- [pythia-160m-minipile_loss-sampled](https://huggingface.co/Marcus2112/pythia-160m-minipile_loss-sampled)
- [pythia-160m-minipile_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_density-proportioned)
- [pythia-160m-minipile_low_density](https://huggingface.co/Marcus2112/pythia-160m-minipile_low-density)
- [pythia-160m-minipile_k440](https://huggingface.co/Marcus2112/pythia-160m-minipile_k440)
- [pythia-160m-minipile_k440_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_k440_density-proportioned)
- [pythia-160m-minipile_k440_inter-density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_k440_inter-density-proportioned)
- [pythia-160m-minipile_k440_high-inter_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_k440_high-inter_density-proportioned)
- [pythia-160m-minipile_tiny_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_tiny_density-proportioned)
- [pythia-160m-minipile_nano_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_nano_density-proportioned)
- [pythia-160m-minipile_pico_density-proportioned](https://huggingface.co/Marcus2112/pythia-160m-minipile_pico_density-proportioned)
- [pythia-160m-minipile_density_pico_epochs](https://huggingface.co/Marcus2112/pythia-160m-minipile_density_pico_epochs)
- [pythia-160m-minipile_density-proportioned_epochs](https://huggingface.co/Marcus2112/pythia-160m-minipile_density-proportioned_epochs)
- [pythia-1.4b-minipile](https://huggingface.co/Marcus2112/pythia-1.4b-minipile)
- [pythia-1.4b-minipile_reproduction](https://huggingface.co/Marcus2112/pythia-1.4b-minipile_reproduction)
- [pythia-1.4b-minipile_density-proportioned](https://huggingface.co/Marcus2112/pythia-1.4b-minipile_density-proportioned)

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
- [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only (Penedo, et al. 2023)](https://arxiv.org/abs/2306.01116)
- [Locality Sensitive Hashing (LSH): The Illustrated Guide (pinecone.io)](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)
- [Hashing for Similarity Search: A Survey (Wang, et al. 2014)](https://arxiv.org/pdf/1408.2927)