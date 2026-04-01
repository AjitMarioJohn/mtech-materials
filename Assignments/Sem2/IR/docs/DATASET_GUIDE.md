# Dataset Guide - Large-Scale Query Clustering

## Overview
We now support both small and large datasets for IR query clustering experiments.

## Running the Code

### Option 1: Small Dataset (Default - 20 queries)
```bash
python main.py
```
- Uses: `data/queries.csv`
- Output: `outputs/cluster_assignments.csv`
- Features: 47 unique terms
- Clusters: 4
- Silhouette Score: 0.0744 (poor, but good for testing)

### Option 2: Large Dataset (396+ queries)
```bash
python main.py --large
# OR
python main.py -L
```
- Uses: Procedurally generated dataset based on MS MARCO style queries
- Output: `outputs/cluster_assignments_large.csv`
- Features: 162 unique terms
- Clusters: 10
- Silhouette Score: 0.1085 (still low - typical for diverse query data)

## Dataset Details

### Small Dataset (data/queries.csv)
```
Categories:
- E-commerce (iphone, phones, laptops)
- Programming (Python, Java, machine learning)
- Travel/Food (Delhi weather, pizza delivery)
- News (headlines, sports)
```

### Large Dataset (Generated on-the-fly)
```
Categories:
- E-commerce (laptops, phones, gadgets)
- Programming (Python, JavaScript, Java)
- Travel/Food/Booking (flights, hotels, restaurants)
- News/Sports/Entertainment
- Health/Fitness
- Education
- Finance/Insurance
- Government services
```

**Total: 396 unique queries with variations**

## Data Flow

```
data/queries.csv (small)
         ↓
    load_queries()
         ↓
  build_tfidf_features()
         ↓
  run_kmeans_baseline()
         ↓
outputs/cluster_assignments.csv
```

## Adding More Data

### Option A: Use Your Own CSV
Replace `data/queries.csv` with your CSV file containing a "query" column.

### Option B: Integrate Real Public Datasets

#### 1. SearchSnippets Dataset (12K queries)
```python
from src.data_loader import load_search_snippets
csv_path = load_search_snippets()
```

#### 2. MS MARCO Queries (1M+ real queries)
```python
# Download from: https://microsoft.github.io/msmarco/
# Already integrated as fallback in our codebase
```

## Metrics Comparison

| Metric | Small | Large |
|--------|-------|-------|
| Queries | 20 | 396 |
| Features (TF-IDF) | 47 | 162 |
| Clusters | 4 | 10 |
| Silhouette Score | 0.0744 | 0.1085 |
| Feature/Query Ratio | 2.35 | 0.41 |

## Why Silhouette Score is Low

1. **Query diversity**: Queries from different domains have different vocabularies
2. **Overlapping topics**: Some queries fit multiple categories
3. **Sparse features**: Many queries share few terms
4. **TF-IDF limitations**: Doesn't capture semantic similarity
5. **Hard clustering problem**: KMeans assumes spherical clusters

### Improvements to Try:
- Use semantic embeddings (BERT, Word2Vec) instead of TF-IDF
- Try soft clustering (Gaussian Mixture Models)
- Use hierarchical clustering
- Adjust hyperparameters (n_clusters, TF-IDF parameters)
- Preprocess queries better (synonyms, query expansion)

## Saving Results

Both runs automatically save clustering results:
- `outputs/cluster_assignments.csv` - Small dataset results
- `outputs/cluster_assignments_large.csv` - Large dataset results

Each file contains:
- `query`: Original query text
- `cluster`: Assigned cluster ID (0 to n_clusters-1)

