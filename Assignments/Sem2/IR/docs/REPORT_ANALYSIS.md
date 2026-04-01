# Report Analysis - Query Clustering Assignment

## Experimental setup
We evaluated the same query dataset under two TF-IDF representations and two clustering algorithms.

### Representations
- **Unigram TF-IDF**: `ngram_range=(1,1)`
- **Unigram + Bigram TF-IDF**: `ngram_range=(1,2)`

### Clustering algorithms
- **KMeans**
- **Hierarchical clustering**

The evaluation used silhouette score for quantitative comparison and top terms for qualitative cluster interpretation.

---

## Main results

### 1) Representation comparison
On the small query dataset, unigram TF-IDF performed better than unigram+bigram TF-IDF.

- `tfidf_unigram` achieved the strongest silhouette scores.
- `tfidf_unigram_bigram` produced lower silhouette values and more fragmented clusters.

This suggests that, for this dataset, single-word vocabulary captured the query structure more reliably than adding bigrams.

### 2) Algorithm comparison
Using the best TF-IDF representation, we compared KMeans and hierarchical clustering.

- **KMeans silhouette**: `0.0744`
- **Hierarchical silhouette**: `0.0814`

Hierarchical clustering performed slightly better, although the difference is small.

---

## Interpretation

### Why the silhouette scores are low
The silhouette values are low because this is a small, mixed-intent query dataset with sparse lexical overlap. TF-IDF is a word-based representation, so it cannot fully capture semantic similarity between queries.

Examples of diverse intents in the dataset include:
- "python list comprehension"
- "java stream map example"
- "best pizza delivery"
- "weather in delhi"

These queries belong to different intent groups, but they still share broad terms such as "best", "today", or "example". This weakens cluster separation and lowers the silhouette score.

### Why hierarchical clustering performed slightly better
Hierarchical clustering is often more flexible than KMeans on small sparse datasets. It does not assume spherical clusters, so it can form groupings that better match the irregular structure of short search queries.

---

## Qualitative cluster interpretation
The KMeans baseline produced the following top terms per cluster:

- Cluster 0: `iphone, 14, price, buy, today`
- Cluster 1: `python, example, loop, list, comprehension`
- Cluster 2: `delhi, today, news, learning, weather`
- Cluster 3: `best, pizza, near, 50000, delivery`

These terms show clear semantic themes even though the silhouette score is modest.

### Interpretable intent groups
- **Shopping / mobile search**
- **Programming / coding queries**
- **News / weather / location-based queries**
- **Food / delivery / shopping queries**

This qualitative result is useful because it shows the clustering is meaningful and not random.

---

## Cluster-size analysis
The cluster-size comparison chart shows that both algorithms create imbalanced clusters, which is expected for natural-language queries.

- KMeans produced one large cluster and several smaller ones.
- Hierarchical clustering also produced uneven cluster sizes, but with slightly different group boundaries.

In search query data, intent categories are rarely perfectly balanced, so some unevenness is natural.

---

## Seed stability analysis
To evaluate cluster stability, KMeans can be rerun with multiple random seeds and compared using Adjusted Rand Index (ARI).

- A higher ARI means the clustering is more consistent across seeds.
- A lower ARI means cluster assignments change more across runs.

For this dataset, stability analysis is useful because TF-IDF features are sparse and the query space is small. Even when the silhouette score is low, a stable ARI across seeds would still show that the algorithm produces repeatable groupings.

---

## Conclusion
The baseline pipeline now supports:

- TF-IDF query representation
- KMeans clustering
- Hierarchical clustering
- Silhouette evaluation
- Top-term cluster labeling
- Representation sweeps
- Algorithm comparison
- Seed stability analysis
- Report-ready charts

### Final takeaway
- **Best representation on this dataset**: unigram TF-IDF
- **Best clustering algorithm on this dataset**: hierarchical clustering
- **Overall quality**: modest quantitatively, but interpretable and suitable for intent grouping

This is a solid baseline for the assignment because it demonstrates controlled comparison, evaluation, and qualitative analysis.

