# Report Tables - Query Clustering Assignment

## Methodology summary
We evaluated query clustering using a TF-IDF representation of the query text and compared multiple configurations.

### Data representation
- **Unigram TF-IDF**: `ngram_range=(1,1)`
- **Unigram + Bigram TF-IDF**: `ngram_range=(1,2)`

### Clustering algorithms
- **KMeans**
- **Hierarchical clustering**

### Evaluation
- **Silhouette score** for cluster quality
- **Top terms per cluster** for qualitative interpretation
- **Adjusted Rand Index (ARI)** for stability across random seeds

---

## Table 1. Representation comparison
The sweep results below summarize the best-performing configuration for each TF-IDF representation.

| Representation | Best n-gram range | Best k | Best silhouette | Cluster sizes |
|---|---:|---:|---:|---|
| TF-IDF unigram | 1,1 | 5 | 0.0530 | 0:18 \| 1:230 \| 2:36 \| 3:76 \| 4:36 |
| TF-IDF unigram + bigram | 1,2 | 5 | 0.0382 | 0:244 \| 1:44 \| 2:36 \| 3:42 \| 4:30 |

**Takeaway:** Unigram TF-IDF performed better than unigram+bigram TF-IDF on this dataset.

---

## Table 2. Algorithm comparison
This table compares the two clustering algorithms using the best TF-IDF setup.

| Algorithm | Representation | n-gram range | n_clusters | Silhouette | Cluster sizes |
|---|---|---:|---:|---:|---|
| KMeans | TF-IDF unigram | 1,1 | 4 | 0.0744 | 0:2 \| 1:3 \| 2:11 \| 3:4 |
| Hierarchical | TF-IDF unigram | 1,1 | 4 | 0.0814 | 0:9 \| 1:4 \| 2:5 \| 3:2 |

**Takeaway:** Hierarchical clustering slightly outperformed KMeans on the small query dataset.

---

## Table 3. Seed stability analysis
KMeans was rerun with multiple random seeds to evaluate stability using ARI against seed 1.

| Seed | Silhouette | ARI vs seed 1 | Cluster sizes |
|---:|---:|---:|---|
| 1 | 0.0732 | 1.0000 | 0:12 \| 1:2 \| 2:2 \| 3:4 |
| 7 | 0.0734 | 0.0792 | 0:2 \| 1:3 \| 2:12 \| 3:3 |
| 21 | 0.0703 | 0.2345 | 0:2 \| 1:3 \| 2:3 \| 3:12 |
| 42 | 0.0744 | 0.4908 | 0:2 \| 1:3 \| 2:11 \| 3:4 |
| 99 | 0.0734 | 0.0792 | 0:12 \| 1:3 \| 2:2 \| 3:3 |

**Takeaway:** Clustering is only moderately stable across seeds, which shows that initialization affects the final grouping on this sparse dataset.

---

## Short interpretation for the report
The experiments show that unigram TF-IDF is the most effective representation among the tested options. Hierarchical clustering provides slightly better cluster quality than KMeans for this dataset, although both methods achieve low silhouette scores overall. The low scores are expected because the query set is small, sparse, and contains mixed intents. Stability analysis confirms that KMeans is sensitive to random initialization, so the final clusters can change noticeably across seeds.

