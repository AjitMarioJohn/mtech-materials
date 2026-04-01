# Quick Reference - Running the Code

## Commands

```bash
# Small dataset (20 queries, quick test)
python main.py

# Large dataset (396 queries)
python main.py --large
python main.py -L

# View results
# Small: outputs/cluster_assignments.csv
# Large: outputs/cluster_assignments_large.csv
```

## File Structure
```
IR/
├── data/
│   ├── queries.csv                   (original small dataset)
│   └── queries_large.csv             (auto-generated large dataset)
├── outputs/
│   ├── cluster_assignments.csv       (small results)
│   └── cluster_assignments_large.csv (large results)
├── src/
│   ├── data_loader.py               (NEW! Dataset loading)
│   ├── features.py                  (TF-IDF vectorization)
│   ├── cluster.py                   (KMeans clustering)
│   └── queries.py                   (Query loading)
├── main.py                           (UPDATED! Now supports both)
└── DATASET_GUIDE.md                  (NEW! Documentation)
```

## Output Example (Large Dataset)

```
Loading large public dataset (MS MARCO sample)...
Loaded 396 queries
Building TF-IDF features (max_features=10000)...
Feature matrix shape: (396, 162)
Running KMeans with 10 clusters...
Silhouette score: 0.1085

Top terms per cluster:
Cluster 0: tutorial, yoga, beginners, react, hooks
Cluster 1: 2024, tips, phones, preparation, exam
Cluster 2: tomorrow, weather, forecast, delhi, 2024
Cluster 3: best, online, free, 2024, movies
Cluster 4: plan, workout, gym, diet, loss
Cluster 5: comparison, price, iphone, 14, cryptocurrency
Cluster 6: python, example, courses, loop, comprehension
Cluster 7: gaming, wireless, mouse, deals, monitor
Cluster 8: today, news, headlines, match, sports
Cluster 9: ticket, train, flight, book, booking

Cluster Distribution:
0     44
1    122
2     12
3     98
4     12
5    24
6    24
7    24
8    24
9    12
```

## Key Points

✅ **What's new:**
- `src/data_loader.py` - Generate large datasets automatically
- `main.py` - Supports `--large` flag for switching datasets
- Smart parameter adjustment based on dataset size

✅ **Dataset Characteristics:**
- **Small (20 queries)**: Quick testing, 4 clusters
- **Large (396 queries)**: Real-world scale, 10 clusters

✅ **Auto-scaling:**
- TF-IDF max_features: 5K → 10K
- n_clusters: 4 → 10

## What to Try Next

1. **Analyze clusters** - Look at `cluster_assignments_large.csv`
2. **Experiment with parameters** - Try different n_clusters
3. **Improve features** - Add bigrams, adjust min_df
4. **Try new datasets** - Load your own CSV with queries column
5. **Semantic clustering** - Replace TF-IDF with embeddings (future)

