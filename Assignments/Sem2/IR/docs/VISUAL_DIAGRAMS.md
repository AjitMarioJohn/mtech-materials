# Visual Diagrams & Architecture

## 1. Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY CLUSTERING SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  INPUT LAYER                PROCESSING LAYER              OUTPUT LAYER       │
│  ───────────────────────────────────────────────────────────────────────────│
│                                                                               │
│  ┌──────────────────┐                                    ┌──────────────────┐
│  │  queries.csv     │                                    │  CSV Output      │
│  │                  │                                    │  query|cluster   │
│  │  20 queries      │──────────┐                         │  ───────────────│
│  │                  │          │                         │  iphone 14  │ 0 │
│  └──────────────────┘          │                         │  python ... │ 1 │
│                                │                         │  ...        │ 2 │
│                                ▼                         └──────────────────┘
│  ┌──────────────────┐     ┌──────────────────┐
│  │  data_loader.py  │     │  queries.py      │
│  │                  │     │                  │
│  │  Generate 396    │────▶│  Load & Clean    │
│  │  large queries   │     │                  │
│  │                  │     └────────┬─────────┘
│  └──────────────────┘              │
│                                   ▼
│                          ┌──────────────────┐
│                          │  features.py     │
│                          │  TF-IDF          │
│                          │                  │
│         List of          │  Text → Numbers  │
│         Strings          │  (20 × 47)       │
│                          │                  │
│                          └────────┬─────────┘
│                                   │
│                                   ▼
│                          ┌──────────────────┐
│                          │  cluster.py      │
│                          │  KMeans          │
│                          │                  │
│         TF-IDF Matrix    │  Group similar   │
│         (20 × 47)        │  queries         │
│                          │                  │
│                          └────────┬─────────┘
│                                   │
│                    ┌──────────────┼──────────────┐
│                    ▼              ▼              ▼
│            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│            │ Labels       │  │ Silhouette   │  │ Top Terms    │
│            │ [0,0,3,2,..] │  │ Score: 0.074 │  │ per Cluster  │
│            └──────────────┘  └──────────────┘  └──────────────┘
│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Transformation Pipeline

```
Step 1: RAW TEXT
─────────────────────────────────────────────
"python tutorial"
"python for loop"
"java example"

                        ▼

Step 2: TOKENIZATION (queries.py)
─────────────────────────────────────────────
["python", "tutorial"]
["python", "for", "loop"]
["java", "example"]

                        ▼

Step 3: STOP WORD REMOVAL (features.py)
─────────────────────────────────────────────
["python", "tutorial"]
["python", "loop"]          ← "for" removed
["java", "example"]

                        ▼

Step 4: BUILD VOCABULARY
─────────────────────────────────────────────
Unique terms = ["python", "tutorial", "loop", "java", "example"]
Index:         [0,        1,          2,     3,     4]

                        ▼

Step 5: COMPUTE TF-IDF SCORES
─────────────────────────────────────────────
             python  tutorial  loop  java  example
"python ..."   0.7     0.7     0.0   0.0    0.0
"python ..."   0.7     0.0     0.7   0.0    0.0
"java ..."     0.0     0.0     0.0   0.7    0.7

                        ▼

Step 6: CREATE MATRIX X
─────────────────────────────────────────────
Sparse Matrix (3, 5)
3 documents (queries)
5 features (unique terms)

Each cell = TF-IDF score (importance)
```

---

## 3. TF-IDF Calculation Example

```
Query: "python python tutorial"

TERM FREQUENCY (TF):
─────────────────────
python:   2/3 = 0.67  (appears 2 times out of 3 words)
tutorial: 1/3 = 0.33  (appears 1 time out of 3 words)

INVERSE DOCUMENT FREQUENCY (IDF):
──────────────────────────────────
Total documents (queries) = 20

For "python":
  Appears in 5 queries
  IDF = log(20/5) = log(4) = 0.602

For "tutorial":
  Appears in 2 queries
  IDF = log(20/2) = log(10) = 1.0

TF-IDF SCORE:
──────────────
python:   TF × IDF = 0.67 × 0.602 = 0.40
tutorial: TF × IDF = 0.33 × 1.0   = 0.33

NORMALIZED:
───────────
python:   0.40 / √(0.40² + 0.33²) = 0.77
tutorial: 0.33 / √(0.40² + 0.33²) = 0.64
```

---

## 4. KMeans Clustering Visualization

```
INITIAL STATE: Random cluster centers
──────────────────────────────────────
    Cluster 0              Cluster 1
         C0●                    C1●
    
    query1 ●                        ● query2
    query3 ●                        ● query4
    
(Far apart - random placement)

ITERATION 1: Assign to nearest center
──────────────────────────────────────
    Cluster 0              Cluster 1
         C0●                    C1●
    
    ●query1                        ●query2
    ●query3      ←(closer)        ●query4
    
(C0 is closer to query1 & 3)
(C1 is closer to query2 & 4)

ITERATION 2: Recalculate centers
──────────────────────────────────────
    Cluster 0              Cluster 1
        C0'●                   C1'●
    
    ●query1                       ●query2
    ●query3                       ●query4
    
(Centers moved to center of their queries)

ITERATION 3: Reassign (if positions changed)
──────────────────────────────────────────────
    Same as iteration 2 (CONVERGED!)
    
(Centers no longer move)
```

---

## 5. Silhouette Score Calculation

```
Query: "python tutorial" (assigned to Cluster 1)

Step 1: Calculate a(i) - Cohesion
────────────────────────────────────
Distance to other queries in Cluster 1:
  Distance to "python for loop":     0.15
  Distance to "java stream":         0.85
  Distance to "machine learning":    0.92

Average distance to same cluster:
a(i) = (0.15 + 0.85 + 0.92) / 3 = 0.64

(Lower is better - closer to own cluster)

Step 2: Calculate b(i) - Separation
─────────────────────────────────────
Nearest cluster is Cluster 0 (shopping)

Distance to queries in Cluster 0:
  Distance to "iphone":              0.95
  Distance to "laptop":              0.91
  Distance to "best phone":          0.88

Average distance to nearest other cluster:
b(i) = (0.95 + 0.91 + 0.88) / 3 = 0.91

(Higher is better - far from other clusters)

Step 3: Calculate silhouette
──────────────────────────────
silhouette = (b(i) - a(i)) / max(a(i), b(i))
           = (0.91 - 0.64) / max(0.64, 0.91)
           = 0.27 / 0.91
           = 0.30

(0 to 1 scale - 0.30 is mediocre)

Step 4: Average all queries
──────────────────────────────
Silhouette score = average of all 20 query silhouette values
                 = 0.0744

(Low because clusters overlap heavily)
```

---

## 6. Module Dependency Graph

```
main.py
├─ imports load_queries() from queries.py
├─ imports load_ms_marco_sample() from data_loader.py
├─ imports build_tfidf_features() from features.py
└─ imports from cluster.py:
   ├─ run_kmeans_baseline()
   ├─ compute_silhouette_score()
   └─ get_top_terms_per_cluster()

queries.py
├─ imports Path from pathlib
└─ imports pd from pandas

data_loader.py
├─ imports Path from pathlib
├─ imports pd from pandas
└─ imports urllib for downloading

features.py
└─ imports TfidfVectorizer from sklearn

cluster.py
├─ imports KMeans from sklearn
└─ imports silhouette_score from sklearn
```

---

## 7. Data Shape Transformations

```
START
  ↓
queries.py:load_queries()
  Input:  CSV file with 20 rows
  Output: List[str] with length 20
  Shape:  (20,)
  ↓
features.py:build_tfidf_features()
  Input:  List[str] with 20 items
  Output: Sparse matrix + Vectorizer
  Shape:  (20, 47)  ← 20 queries, 47 unique terms
  ↓
cluster.py:run_kmeans_baseline()
  Input:  Sparse matrix (20, 47)
  Output: Labels array + KMeans model
  Shape:  (20,)  ← One label per query
  Values: [0, 0, 3, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 3]
  ↓
cluster.py:compute_silhouette_score()
  Input:  Matrix (20, 47) + Labels (20,)
  Output: Single float number
  Value:  0.0744
  ↓
cluster.py:get_top_terms_per_cluster()
  Input:  KMeans model + Vectorizer
  Output: Dict[cluster_id → List[word_names]]
  Value:  {
            0: ['iphone', '14', 'price', 'buy', 'today'],
            1: ['python', 'example', 'loop', 'list', 'comprehension'],
            2: ['delhi', 'today', 'news', 'learning', 'weather'],
            3: ['best', 'pizza', 'near', '50000', 'delivery'],
          }
  ↓
main.py:Save to CSV
  Input:  queries(20,) + labels(20,)
  Output: CSV file with 20 rows × 2 columns
  File:   outputs/cluster_assignments.csv
```

---

## 8. Information Flow Diagram

```
┌──────────────┐
│ Raw Queries  │
│ (CSV File)   │
└──────┬───────┘
       │
       │ [Text data]
       ▼
┌──────────────────┐
│  Load Queries    │ ← queries.py
│  - Check exists  │
│  - Parse CSV     │
│  - Clean data    │
└──────┬───────────┘
       │
       │ [List of strings]
       │ ["python tutorial", "java code", ...]
       │
       ▼
┌──────────────────┐
│ Build TF-IDF     │ ← features.py
│ - Tokenize       │
│ - Remove stops   │
│ - Vectorize      │
└──────┬───────────┘
       │
       │ [Sparse matrix]
       │ (20 queries × 47 terms)
       │
       ├─────────────────┬──────────────────┐
       │                 │                  │
       ▼                 ▼                  ▼
   ┌────────┐    ┌──────────────┐  ┌─────────────┐
   │ KMeans │    │  Silhouette  │  │  Top Terms  │
   │ (label)│    │  (evaluate)  │  │  (explain)  │
   └────┬───┘    └──────┬───────┘  └────────┬────┘
        │                │                  │
        └────────┬───────┴──────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Save Results   │
        │  (CSV File)     │
        └─────────────────┘
```

---

## 9. Comparison: Small vs Large Dataset

```
                SMALL DATASET        LARGE DATASET
                ─────────────        ─────────────
Queries         20                   396
Features        47                   162
Clusters        4                    10
Feature/Query   2.35                 0.41

TF-IDF Params:
  max_features  5000                 10000
  min_df        1                    1

KMeans Params:
  n_clusters    4                    10
  random_state  42                   42
  n_init        10                   10

Output Files:
  Small         cluster_assignments.csv
  Large         cluster_assignments_large.csv

Silhouette:     0.0744               0.1085
Performance:    < 1 sec              2-3 sec
```

---

## 10. Error Handling Flow

```
main.py ────────────────────┐
                            │
Load Dataset ──────────────▶│
  ├─ Check file exists      │
  ├─ Check 'query' column   │
  ├─ Remove empties         │
  └─ Return list or ERROR ──┤──┬─ FileNotFoundError
                            │  ├─ ValueError
                            │  └─ Continue
                            │
Build Features ────────────▶│
  ├─ Check not empty        │
  ├─ Tokenize & clean       │
  └─ Vectorize or ERROR ────┤──┬─ ValueError
                            │  └─ Continue
                            │
Run KMeans ────────────────▶│
  ├─ Check n_clusters >= 2  │
  ├─ Fit model              │
  └─ Return or ERROR ────────┤──┬─ ValueError
                            │  └─ Continue
                            │
Get Top Terms ─────────────▶│
  ├─ min(requested, available)  [FIX!]
  ├─ Extract terms          │
  └─ Return dict ────────────┤──✓ No error
                            │
                            ▼
                    SUCCESS - PRINT & SAVE
```

---

## 11. Memory Usage Diagram

```
Memory Usage During Execution
──────────────────────────────

queries list (20 items):
  "python tutorial"              ≈ 0.005 MB
  "buy iphone 14"                ≈ 0.005 MB
  ... × 20                       ≈ 0.1 MB total

TF-IDF Matrix X (20 × 47):
  Sparse matrix (only non-zero): ≈ 0.2 MB
  Dense would be:                ≈ 0.04 MB

Vectorizer object:
  Vocabulary (47 terms):         ≈ 0.01 MB

KMeans model:
  Cluster centers (4 × 47):      ≈ 0.002 MB
  Labels (20,):                  ≈ 0.0002 MB

Output DataFrame:
  20 queries + 20 labels:        ≈ 0.01 MB

TOTAL MEMORY: ≈ 0.4 MB (very small!)

For large dataset (396 × 162):
  TF-IDF Matrix:                 ≈ 1.2 MB
  TOTAL:                         ≈ 2 MB
```

---

## 12. Execution Timeline

```
python main.py
    │
    ▼ (1-2 ms)
main() starts
    │
    ├─▶ load_queries()          (1-2 ms)
    │   └─ Read CSV
    │
    ├─▶ build_tfidf_features()  (10-20 ms)
    │   └─ Vectorize 20 texts
    │
    ├─▶ run_kmeans_baseline()   (5-10 ms)
    │   └─ Fit KMeans
    │
    ├─▶ compute_silhouette()    (5-10 ms)
    │   └─ Calculate score
    │
    ├─▶ get_top_terms()         (1-2 ms)
    │   └─ Extract top words
    │
    └─▶ Save & Print            (2-5 ms)
        └─ Write CSV
    │
    ▼ (≈30-50 ms total)
Done!
```


