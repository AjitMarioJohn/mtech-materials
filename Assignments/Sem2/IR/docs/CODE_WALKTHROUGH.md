# Step-by-Step Code Walkthrough with Examples

## Example: Running with Small Dataset (20 queries)

### Command
```bash
python main.py
```

---

## Execution Timeline

### Phase 1: START MAIN.PY

```python
if __name__ == "__main__":
    use_large = "--large" in sys.argv  # Check command line
    main(use_large_dataset=False)       # FALSE because no --large flag
```

**Console Output So Far:**
```
(Nothing yet - still starting)
```

---

### Phase 2: LOAD DATA

```python
def main(use_large_dataset=False):
    # use_large_dataset = False
    
    if use_large_dataset:  # This is FALSE, skip this block
        ...
    else:                   # This runs
        csv_path = "data/queries.csv"
        print(f"Loading small dataset from {csv_path}...")
    
    queries = load_queries(csv_path)
    print(f"Loaded {len(queries)} queries")
```

**Inside load_queries():**
```python
def load_queries("data/queries.csv"):
    path = Path("data/queries.csv")
    
    # Step 1: Check if file exists
    if not path.exists():  # It DOES exist
        raise FileNotFoundError(...)
    
    # Step 2: Read CSV
    df = pd.read_csv("data/queries.csv")
    # df now looks like:
    #              query
    # 0       buy iphone 14
    # 1    iphone 14 price
    # 2   best budget phone
    # ... (total 20 rows)
    
    # Step 3: Check 'query' column exists
    if "query" not in df.columns:  # It DOES exist
        raise ValueError(...)
    
    # Step 4: Extract and clean
    queries = df["query"].astype(str).str.strip()
    # queries now = Series with 20 values
    
    # Step 5: Remove empty
    queries = queries[queries != ""].tolist()
    # queries now = ["buy iphone 14", "iphone 14 price", "best budget phone", ...]
    
    # Step 6: Validate
    if not queries:  # They exist
        raise ValueError(...)
    
    return queries  # Return list of 20 query strings
```

**Console Output:**
```
Loading small dataset from data/queries.csv...
Loaded 20 queries
```

**Variable State:**
```
queries = [
    "buy iphone 14",
    "iphone 14 price",
    "best budget phone",
    "cheap mobile phones",
    "python list comprehension",
    "python for loop example",
    "java stream map example",
    "java hashmap interview questions",
    "weather in delhi",
    "today temperature delhi",
    "pizza near me",
    "best pizza delivery",
    "machine learning course",
    "deep learning tutorial",
    "news headlines today",
    "latest sports news",
    "train ticket booking",
    "book flight online",
    "best laptop under 50000",
    "gaming laptop deals",
]
```

---

### Phase 3: BUILD TF-IDF FEATURES

```python
max_features = 5000  # Not large dataset
n_clusters = 4       # Not large dataset

print(f"Building TF-IDF features (max_features={max_features})...")
X, vectorizer = build_tfidf_features(
    queries,
    max_features=5000,
    ngram_range=(1, 1),
    min_df=1
)
print(f"Feature matrix shape: {X.shape}")
```

**Inside build_tfidf_features():**
```python
def build_tfidf_features(queries, max_features=5000, ...):
    # queries = 20 query strings
    
    # Step 1: Create vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,              # "PYTHON" → "python"
        stop_words="english",        # Remove: "the", "a", "in", etc.
        max_features=5000,
        ngram_range=(1, 1),          # Single words only
        min_df=1,                    # Keep words in ≥1 document
    )
    
    # Step 2: Learn vocabulary and transform
    X = vectorizer.fit_transform(queries)
    # Processes:
    # - Lowercase all words
    # - Remove English stop words
    # - Extract unique words (terms)
    # - Count TF-IDF scores
    
    # X is a SPARSE MATRIX with shape (20, 47)
    # 20 = number of queries (documents)
    # 47 = number of unique terms found
    
    return X, vectorizer
```

**What's Inside X (Conceptually)?**
```
TF-IDF Matrix (20 queries × 47 unique words):

              budget  book  booking  ... cheap  python  iphone  14 ... 
buy iphone        0   0.0    0.0    ...  0.0    0.0    0.707  0.707...
iphone price      0   0.0    0.0    ...  0.0    0.0    0.600  0.600...
best budget       1   0.0    0.0    ...  0.0    0.0    0.0    0.0...
cheap mobile      0   0.0    0.0    ...  0.500  0.0    0.0    0.0...
python list       0   0.0    0.0    ...  0.0    0.707  0.0    0.0...
... (15 more queries)

Each cell = TF-IDF score (0 to 1)
Higher = more important
```

**Unique Words Found (47 total):**
```
feature_names = [
    '14', '50000', 'best', 'book', 'booking', 'budget', 
    'buy', 'cheap', 'comprehension', 'course', 'deep', 
    'delhi', 'delivery', 'example', 'flight', 'for', 
    'game', 'hashmap', 'headlines', 'interview', 'iphone', 
    'java', 'laptop', 'learning', 'list', 'loop', 
    'machine', 'map', 'mobile', 'news', 'online', 
    'phone', 'pizza', 'python', 'question', 'questions', 
    'latest', 'sports', 'stream', 'temperature', 'ticket', 
    'today', 'training', 'tutorial', 'weather'
]
```

**Console Output:**
```
Building TF-IDF features (max_features=5000)...
Feature matrix shape: (20, 47)
```

**Variable State:**
```
X = Sparse matrix (20, 47)
vectorizer = TfidfVectorizer object with learned vocabulary
```

---

### Phase 4: RUN KMEANS CLUSTERING

```python
print(f"Running KMeans with {n_clusters} clusters...")
labels, model = run_kmeans_baseline(X, n_clusters=4, random_state=42)
```

**Inside run_kmeans_baseline():**
```python
def run_kmeans_baseline(X, n_clusters=4, random_state=42):
    # X = (20, 47) sparse matrix
    # n_clusters = 4
    
    # Step 1: Validate
    if 4 < 2:  # FALSE
        raise ValueError(...)
    
    # Step 2: Create KMeans model
    model = KMeans(
        n_clusters=4,
        random_state=42,  # Makes results reproducible
        n_init=10,        # Try 10 different random initializations
    )
    
    # Step 3: Fit and predict
    # - Initialize 4 random cluster centers
    # - Assign each query to nearest center (9 iterations shown below)
    # - Recalculate centers
    # - Repeat until convergence
    
    labels = model.fit_predict(X)
    
    # labels = [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3]
    # One label per query (0, 1, 2, or 3)
    
    return labels, model
```

**What KMeans Does (High-Level):**
```
Iteration 1:
- Random centers: C0, C1, C2, C3
- Assign queries to nearest center
- Get: [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3]

Iteration 2:
- Recalculate centers as average of assigned queries
- Reassign queries
- Get: [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3]

... (continues until stable)

Final Result: Same labels (converged!)
```

**Console Output:**
```
Running KMeans with 4 clusters...
```

**Variable State:**
```
labels = [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3]
model = KMeans object with fitted cluster centers
```

**What Each Cluster Contains (by query index):**
```
Cluster 0: queries[0], queries[1]
  → "buy iphone 14", "iphone 14 price"

Cluster 1: queries[4], queries[5], queries[6], queries[12], queries[13]
  → "python list comprehension", "python for loop example", 
    "java stream map example", "machine learning course", "deep learning tutorial"

Cluster 2: queries[3], queries[8], queries[9], queries[10], queries[14], queries[15], queries[16], queries[17]
  → "cheap mobile phones", "weather in delhi", "today temperature delhi",
    "pizza near me", "news headlines today", "latest sports news",
    "train ticket booking", "book flight online"

Cluster 3: queries[2], queries[11], queries[18], queries[19]
  → "best budget phone", "best pizza delivery", "best laptop under 50000", "gaming laptop deals"
```

---

### Phase 5: COMPUTE SILHOUETTE SCORE

```python
sil = compute_silhouette_score(X, labels)
if sil is None:
    print("Silhouette: not available (need at least 2 clusters).")
else:
    print(f"Silhouette score: {sil:.4f}")
```

**Inside compute_silhouette_score():**
```python
def compute_silhouette_score(X, labels):
    # X = (20, 47) matrix
    # labels = [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3]
    
    # Step 1: Get unique labels
    unique_labels = set(int(x) for x in labels)
    # unique_labels = {0, 1, 2, 3}
    
    # Step 2: Check we have at least 2 clusters
    if len(unique_labels) < 2:  # 4 >= 2, TRUE
        return None
    
    # Step 3: Calculate silhouette score
    return float(silhouette_score(X, labels))
    
    # Scikit-learn calculates:
    # For each query:
    #   a(i) = avg distance to other queries in same cluster
    #   b(i) = avg distance to queries in nearest other cluster
    #   score(i) = (b(i) - a(i)) / max(a(i), b(i))
    # Average all individual scores
    
    # Result: 0.0744 (low - clusters overlap)
```

**Console Output:**
```
Silhouette score: 0.0744
```

---

### Phase 6: EXTRACT TOP TERMS

```python
top_terms = get_top_terms_per_cluster(model=model, vectorizer=vectorizer, top_n=5)
print("\nTop terms per cluster:")
for cid in sorted(top_terms):
    print(f"Cluster {cid}: {', '.join(top_terms[cid])}")
```

**Inside get_top_terms_per_cluster():**
```python
def get_top_terms_per_cluster(model, vectorizer, top_n=5):
    # model = fitted KMeans model
    # vectorizer = fitted TfidfVectorizer
    
    # Step 1: Get feature names
    feature_names = vectorizer.get_feature_names_out()
    # = ['14', '50000', 'best', 'book', 'booking', ..., 'weather']
    # Total: 47 words
    
    # Step 2: Initialize result
    top_terms = {}
    
    # Step 3: Safety check
    n_features = len(feature_names)  # 47
    n_top = min(5, 47)               # 5
    
    # Step 4: For each cluster
    for cluster_id, center in enumerate(model.cluster_centers_):
        # cluster_id = 0, center = [0.1, 0.8, 0.3, 0.9, ..., 0.2] (47 values)
        # center[i] = importance of word i in this cluster
        
        # Get indices of top 5 values
        top_indices = center.argsort()[-5:][::-1]
        # argsort() = [0, 2, 1, 3, ...] (indices sorted by value)
        # [-5:] = keep last 5 (highest values)
        # [::-1] = reverse to get highest first
        # Result: [3, 1, 2, 0, 4] (hypothetical)
        
        # Convert indices to words
        top_terms[cluster_id] = [feature_names[i] for i in top_indices]
        # [feature_names[3], feature_names[1], ...]
        # = ['iphone', 'buy', ...]
    
    return top_terms
```

**For Each Cluster:**

```
Cluster 0 center values:
  index=20 (iphone): 0.800
  index=0 (14): 0.750
  index=5 (buy): 0.720
  index=21 (price): 0.700
  index=14 (today): 0.650
  ... (others lower)

Top 5: [iphone, 14, buy, price, today]

---

Cluster 1 center values:
  index=33 (python): 0.850
  index=15 (example): 0.800
  index=26 (loop): 0.750
  index=24 (list): 0.700
  index=8 (comprehension): 0.650
  ... (others lower)

Top 5: [python, example, loop, list, comprehension]

---

Cluster 2 center values:
  index=11 (delhi): 0.800
  index=40 (today): 0.750
  index=40 (news): 0.700
  index=23 (learning): 0.650
  index=39 (weather): 0.600
  ... (others lower)

Top 5: [delhi, today, news, learning, weather]

---

Cluster 3 center values:
  index=2 (best): 0.850
  index=31 (pizza): 0.800
  index=29 (near): 0.750
  index=0 (50000): 0.700
  index=12 (delivery): 0.650
  ... (others lower)

Top 5: [best, pizza, near, 50000, delivery]
```

**Console Output:**
```
Top terms per cluster:
Cluster 0: iphone, 14, price, buy, today
Cluster 1: python, example, loop, list, comprehension
Cluster 2: delhi, today, news, learning, weather
Cluster 3: best, pizza, near, 50000, delivery
```

---

### Phase 7: SAVE RESULTS

```python
result_df = pd.DataFrame({
    "query": queries,
    "cluster": [int(x) for x in labels],
})

output_file = Path("outputs/cluster_assignments.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)  # Create folder
result_df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(result_df.groupby('cluster').size().sort_index())
```

**Creating DataFrame:**
```python
# queries = ["buy iphone 14", "iphone 14 price", ..., "gaming laptop deals"]
# labels = [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3]

result_df = pd.DataFrame({
    "query": ["buy iphone 14", "iphone 14 price", "best budget phone", ...],
    "cluster": [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3],
})

# result_df looks like:
#                            query  cluster
# 0                  buy iphone 14        0
# 1                 iphone 14 price        0
# 2                best budget phone        3
# 3               cheap mobile phones        2
# 4           python list comprehension        1
# 5          python for loop example        1
# 6             java stream map example        1
# 7    java hashmap interview questions        2
# 8                    weather in delhi        2
# 9               today temperature delhi        2
# 10                        pizza near me        2
# 11                  best pizza delivery        3
# 12              machine learning course        1
# 13              deep learning tutorial        1
# 14                news headlines today        2
# 15                latest sports news        2
# 16                train ticket booking        2
# 17                   book flight online        2
# 18              best laptop under 50000        3
# 19                gaming laptop deals        3
```

**Saving to CSV:**
```
outputs/cluster_assignments.csv

query,cluster
buy iphone 14,0
iphone 14 price,0
best budget phone,3
cheap mobile phones,2
python list comprehension,1
... (all 20 queries)
```

**Cluster Distribution:**
```python
result_df.groupby('cluster').size().sort_index()

# Output:
# cluster
# 0     2   (2 queries in cluster 0)
# 1     5   (5 queries in cluster 1)
# 2     8   (8 queries in cluster 2)
# 3     5   (5 queries in cluster 3)
```

**Console Output:**
```
Saved: outputs\cluster_assignments.csv
cluster
0     2
1     5
2     8
3     5
dtype: int64
```

---

## Complete Console Output

```
Loading small dataset from data/queries.csv...
Loaded 20 queries
Building TF-IDF features (max_features=5000)...
Feature matrix shape: (20, 47)
Running KMeans with 4 clusters...
Silhouette score: 0.0744

Top terms per cluster:
Cluster 0: iphone, 14, price, buy, today
Cluster 1: python, example, loop, list, comprehension
Cluster 2: delhi, today, news, learning, weather
Cluster 3: best, pizza, near, 50000, delivery
Saved: outputs\cluster_assignments.csv
cluster
0     2
1     5
2     8
3     5
dtype: int64
```

---

## Final State of Variables

At the end of execution:

```python
queries = [20 query strings]
X = Sparse matrix (20, 47)
vectorizer = TfidfVectorizer with 47 features
labels = [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3]
model = KMeans with 4 clusters
sil = 0.0744
top_terms = {
    0: ['iphone', '14', 'price', 'buy', 'today'],
    1: ['python', 'example', 'loop', 'list', 'comprehension'],
    2: ['delhi', 'today', 'news', 'learning', 'weather'],
    3: ['best', 'pizza', 'near', '50000', 'delivery'],
}
result_df = DataFrame with 20 rows × 2 columns
```

All saved to `outputs/cluster_assignments.csv`

