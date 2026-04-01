# Code Understanding Quiz & Exercises

## Quiz 1: Basic Understanding

### Question 1.1: What does `queries.py` do?
**Answer:** 
- Reads queries from a CSV file
- Cleans the data (removes empty entries, trims whitespace)
- Returns a list of query strings
- Validates that the 'query' column exists

### Question 1.2: What is TF-IDF?
**Answer:**
- TF = Term Frequency (how often a word appears in a query)
- IDF = Inverse Document Frequency (how rare a word is across all queries)
- TF-IDF combines both: important words × rare words = high score
- Used to convert text into numerical features

### Question 1.3: What does KMeans do?
**Answer:**
- Groups similar queries into clusters
- Works by creating cluster centers and iteratively:
  1. Assign queries to nearest center
  2. Recalculate centers as average of queries
  3. Repeat until convergence

### Question 1.4: What is a silhouette score?
**Answer:**
- Measures clustering quality (0 to 1, higher is better)
- Compares:
  - How close queries are to their own cluster
  - How far queries are from other clusters
- Low score (0.07) = clusters overlap significantly

---

## Quiz 2: Code Tracing

### Question 2.1: Trace this code snippet

```python
queries = ["python tutorial", "java code", "python example"]
X, vectorizer = build_tfidf_features(queries, max_features=5000)
print(X.shape)
```

**What will be printed?**

<details>
<summary>Answer</summary>

```
(3, 5)
```

**Explanation:**
- 3 = number of queries (documents)
- 5 = number of unique terms
  - "python" (appears in 2 queries)
  - "tutorial"
  - "java"
  - "code"
  - "example"

Note: "the", "a", etc. are removed by stop_words="english"
</details>

---

### Question 2.2: Trace this execution

```python
labels = [0, 0, 1, 2, 2, 1, 1]
clusters_per_label = {}

for i, label in enumerate(labels):
    if label not in clusters_per_label:
        clusters_per_label[label] = []
    clusters_per_label[label].append(i)

print(clusters_per_label)
```

**What will be printed?**

<details>
<summary>Answer</summary>

```python
{0: [0, 1], 1: [2, 5, 6], 2: [3, 4]}
```

**Explanation:**
- Label 0 appears at indices 0, 1
- Label 1 appears at indices 2, 5, 6
- Label 2 appears at indices 3, 4
</details>

---

### Question 2.3: What's wrong with this code?

```python
def get_top_terms(model, vectorizer, top_n=100):
    feature_names = vectorizer.get_feature_names_out()  # Only 47 features
    
    for cluster_id, center in enumerate(model.cluster_centers_):
        # Trying to get top 100 from 47 features!
        top_indices = center.argsort()[-top_n:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        print(f"Cluster {cluster_id}: {top_terms}")
```

<details>
<summary>Answer</summary>

**Error:** IndexError: index out of bounds

**Why:** 
- Requesting top 100 terms but only 47 features exist
- `top_indices` will include indices >= 47
- `feature_names[i]` will fail when i >= 47

**Fix:**
```python
n_top = min(top_n, len(feature_names))
top_indices = center.argsort()[-n_top:][::-1]
```

This is exactly what we fixed in your code!
</details>

---

## Quiz 3: Design Decisions

### Question 3.1: Why use sparse matrix for TF-IDF?

<details>
<summary>Answer</summary>

**Reason:** Memory efficiency

**Example:**
- Dense matrix (20 × 47):
  ```
  [0.5, 0.0, 0.0, 0.8, 0.0, ..., 0.0]  ← Many zeros!
  ```
  Stores all 47 values per query

- Sparse matrix:
  ```
  {0: 0.5, 3: 0.8}  ← Only non-zero values
  ```
  Stores only 2 values per query

**Why most queries are sparse:**
- Each query mentions only a few topics
- Most term-query combinations don't appear
- Sparse format saves 90%+ memory
</details>

---

### Question 3.2: Why does `data_loader.py` create query variations?

<details>
<summary>Answer</summary>

**Reasons:**
1. **Increase dataset size:** 1 sample → 6 variations = more data
2. **Create realistic variation:** Real users search different ways:
   - "python tutorial"
   - "python tutorial 2024"
   - "python tutorial online"
   - "how to python tutorial"
3. **Better clustering:** More queries = better cluster definition
4. **Simulate synonymy:** Variations capture related queries

**Example:**
```
Original: "buy iphone 14"
Variations:
  1. "buy iphone 14"
  2. "buy iphone 14 2024"
  3. "best buy iphone 14"
  4. "buy iphone 14 online"
  5. "how to buy iphone 14"
  6. "buy iphone 14 tutorial"
```

This makes the dataset more realistic without external data sources.
</details>

---

### Question 3.3: Why increase n_clusters from 4 to 10 for large dataset?

<details>
<summary>Answer</summary>

**Reasons:**
1. **More data = more natural clusters:** 20 queries in 4 clusters is forced
   - 396 queries in 10 clusters = ~40 per cluster = more natural

2. **Diminishing returns:** More clusters on small data just fragments
   - Small dataset: 4 clusters are enough
   - Large dataset: 10 clusters capture more granularity

3. **Data diversity:** Large dataset has 8+ different domains
   - Shopping, Programming, Food, News, Health, Education, Finance, etc.
   - 4 clusters can't capture this
   - 10 clusters provides better separation

4. **Rule of thumb:** √(n_queries / 2)
   - Small: √(20/2) ≈ 3-4 ✓
   - Large: √(396/2) ≈ 14 (we use 10 for safety)
</details>

---

## Quiz 4: Output Interpretation

### Question 4.1: Interpret these results

```
Cluster 0: iphone, 14, price, buy, today
Cluster 1: python, example, loop, list, comprehension
Cluster 2: delhi, today, news, learning, weather
Cluster 3: best, pizza, near, 50000, delivery
```

**What is each cluster about?**

<details>
<summary>Answer</summary>

- **Cluster 0:** Mobile shopping (iphone queries)
- **Cluster 1:** Programming tutorials (python queries)
- **Cluster 2:** News & weather (delhi location queries)
- **Cluster 3:** Restaurant/food search (pizza, delivery)

**Quality assessment:**
- ✅ Clear semantic separation
- ✅ Each cluster has a topic
- ⚠️ "today" appears in multiple clusters (ambiguous)
- ⚠️ Silhouette score is low (0.07) - clusters overlap
</details>

---

### Question 4.2: What does this silhouette score mean?

```
Silhouette score: 0.0744
```

**Is this good or bad?**

<details>
<summary>Answer</summary>

**Rating:** POOR

**Scale:**
- 0.7-1.0: Excellent ✓✓✓
- 0.5-0.7: Good ✓✓
- 0.3-0.5: Fair ⚠️
- 0.0-0.3: Poor ❌
- < 0.0: Very bad ❌❌

**What it means:**
- Queries are **not well separated** by topic
- Some queries could fit in multiple clusters
- Clusters **heavily overlap**
- KMeans struggles with this data

**Why so low:**
1. TF-IDF only looks at word frequency
2. Different domains share few words
3. Some queries are genuinely ambiguous
4. Dataset is too diverse for simple word-based clustering

**Improvement strategies:**
- Use semantic embeddings (BERT, Word2Vec)
- Use soft clustering (Gaussian Mixture Models)
- Preprocess better (lemmatization, synonyms)
- Try hierarchical clustering
</details>

---

## Exercises

### Exercise 1: Modify the code to extract top 3 terms instead of 5

**Task:** Update `main.py` to call `get_top_terms_per_cluster()` with `top_n=3`

<details>
<summary>Solution</summary>

```python
# Change this line:
top_terms = get_top_terms_per_cluster(model=model, vectorizer=vectorizer, top_n=5)

# To this:
top_terms = get_top_terms_per_cluster(model=model, vectorizer=vectorizer, top_n=3)
```

**Expected output:**
```
Cluster 0: iphone, 14, price
Cluster 1: python, example, loop
Cluster 2: delhi, today, news
Cluster 3: best, pizza, near
```
</details>

---

### Exercise 2: Change the number of clusters

**Task:** Try clustering with 6 clusters instead of 4 (small dataset)

<details>
<summary>Solution</summary>

```python
# In main.py, change:
n_clusters = 4 if not use_large_dataset else 10

# To:
n_clusters = 6 if not use_large_dataset else 10

# Then run:
# python main.py
```

**What to observe:**
- Silhouette score will likely increase (more clusters = better fit)
- Cluster sizes will be smaller
- Some clusters might have only 2-3 queries
- Check if the clustering makes semantic sense
</details>

---

### Exercise 3: Add a new feature - cluster validation

**Task:** Add a function that counts how many queries are in each cluster

<details>
<summary>Solution</summary>

```python
def analyze_clusters(queries, labels):
    """Analyze cluster distribution"""
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1
    
    return cluster_counts

# In main.py, add after saving:
cluster_stats = analyze_clusters(queries, labels)
print("\nCluster Statistics:")
for cluster_id in sorted(cluster_stats.keys()):
    count = cluster_stats[cluster_id]
    percentage = (count / len(queries)) * 100
    print(f"Cluster {cluster_id}: {count} queries ({percentage:.1f}%)")
```

**Expected output:**
```
Cluster Statistics:
Cluster 0: 2 queries (10.0%)
Cluster 1: 5 queries (25.0%)
Cluster 2: 8 queries (40.0%)
Cluster 3: 5 queries (25.0%)
```
</details>

---

### Exercise 4: Add a new feature - query-to-cluster mapping

**Task:** Show which queries are in each cluster

<details>
<summary>Solution</summary>

```python
def print_cluster_contents(queries, labels):
    """Print all queries in each cluster"""
    clusters = {}
    for query, label in zip(queries, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(query)
    
    for cluster_id in sorted(clusters.keys()):
        print(f"\nCluster {cluster_id}:")
        for query in clusters[cluster_id]:
            print(f"  - {query}")

# In main.py, add after printing top terms:
print("\nCluster Contents:")
print_cluster_contents(queries, labels)
```

**Expected output:**
```
Cluster Contents:

Cluster 0:
  - buy iphone 14
  - iphone 14 price

Cluster 1:
  - python list comprehension
  - python for loop example
  - java stream map example
  - machine learning course
  - deep learning tutorial

... (more clusters)
```
</details>

---

## Knowledge Check Quiz

**Complete these sentences:**

1. TF-IDF converts **______** into **______** for clustering
   <details><summary>Answer</summary>text, numbers</details>

2. KMeans groups queries by finding **______** distance
   <details><summary>Answer</summary>minimum (or nearest)</details>

3. Silhouette score measures **______** of clustering
   <details><summary>Answer</summary>quality (or goodness/validity)</details>

4. A sparse matrix stores only **______** values
   <details><summary>Answer</summary>non-zero</details>

5. Stop words like "the" are removed to **______** noise
   <details><summary>Answer</summary>reduce</details>

6. The vectorizer learns a **______** of unique terms
   <details><summary>Answer</summary>vocabulary</details>

7. KMeans converges when cluster centers **______**
   <details><summary>Answer</summary>stop moving (or don't change)</details>

8. We use `min(top_n, n_features)` to prevent **______**
   <details><summary>Answer</summary>IndexError (or accessing invalid indices)</details>

---

## Summary of Key Concepts

✅ **Text Processing:**
- Tokenization: Break text into words
- Stop word removal: Remove common words
- Vectorization: Convert words to numbers

✅ **TF-IDF:**
- Combines word frequency and rarity
- Captures local (TF) and global (IDF) importance
- Creates sparse numerical representations

✅ **KMeans:**
- Unsupervised clustering algorithm
- Minimizes within-cluster distance
- Iterative process until convergence

✅ **Evaluation:**
- Silhouette score: Measure clustering quality
- Cluster size: Check balance
- Top terms: Interpret clusters

✅ **Error Handling:**
- Check data exists
- Validate shapes/sizes
- Handle edge cases (sparse matrices)

---

## Next Learning Steps

When you're ready:
1. Try semantic embeddings (BERT/Word2Vec)
2. Experiment with other clustering algorithms (hierarchical, DBSCAN)
3. Implement cross-validation for parameter tuning
4. Try hierarchical agglomerative clustering
5. Learn about topic modeling (LDA, NMF)

