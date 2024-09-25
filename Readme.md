# Miniminter

Reproducing MiniPile from 'The Pile' from scratch using HuggingFace and Python.

## Roadmap

- Obtain the deduplicated "The Pile" dataset. The authors state that their starting point is the deduplicated "The Pile" dump published by EleutherAI on the HF Hub.

- Extract embeddings for each document in "The Pile". Use the E5-Large model (Embeddings from bidirectional Encoder representations) to derive embeddings.

- Perform k-means clustering on the document embeddings. Use batch k-means clustering with cosine distance between normalized embeddings.
    - Set k to 220, i.e., 10 clusters per subset of "The Pile", and 
    - use a batch size of 16384.

- Manually review and curate the clusters. For each cluster, sort the documents by their distance to the assigned centroid. Assess the quality of data in a cluster based on the five examples closest to and furthest from the centroid. Exclude clusters containing undesirable categories, such as near-duplicate documents, pornography, website navigation bars, product specifications, and long lists of named entities.

- Create MiniPile by randomly selecting documents from the remaining clusters. Instead of selecting only the documents closest to the centroid in each cluster, randomly select documents from each cluster. This selection method led to better GLUE results in preliminary BERT training runs. The final MiniPile dataset should include 1 million training examples, 500 validation examples, and 10,000 test examples.


## Related Work

- [HuggingFace: EleutherAI/the-pile](https://huggingface.co/datasets/EleutherAI/pile)
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)
- [The MiniPile Challenge for Data-Efficient Language Models](https://arxiv.org/abs/2304.08442)