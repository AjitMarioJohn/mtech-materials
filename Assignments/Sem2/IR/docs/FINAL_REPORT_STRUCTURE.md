# Final Report Structure (Assignment 1)
Use this structure directly for your PDF report.
## 1. Title and Team Details
- Course: Information Retrieval (S2-25_AIMLZG537)
- Assignment: Query Clustering for Intent Discovery in Web Search
- Group number and member details
## 2. Problem Statement
- What problem is being solved (intent discovery from queries)
- Why clustering helps for grouping similar search intents
## 3. Dataset and Preprocessing
- Dataset used:
  - Small: `data/queries.csv`
  - Large/public track: `data/queries_large.csv`
- Preprocessing summary:
  - lowercasing
  - stop-word removal
  - TF-IDF vectorization
## 4. Methods
### 4.1 Query Representations
- TF-IDF unigram (`ngram_range=(1,1)`)
- TF-IDF unigram+bigram (`ngram_range=(1,2)`)
### 4.2 Clustering Algorithms
- KMeans
- Hierarchical clustering
### 4.3 Evaluation Metrics
- Silhouette score
- ARI for seed stability
- Qualitative cluster interpretation using top terms and pattern labels
## 5. Experiments
### 5.1 Representation Sweep
- Show table from `outputs/experiment_summary.csv`
- Show best-by-representation from `outputs/representation_best_summary.csv`
### 5.2 Algorithm Comparison
- Show table from `outputs/algorithm_comparison.csv`
- Show labels from `outputs/cluster_labels_comparison.csv`
### 5.3 Stability Analysis
- Show table from `outputs/seed_stability_summary.csv`
- Comment on mean ARI and sensitivity to random seed
## 6. Visualizations
Insert figures:
- `outputs/silhouette_comparison.png`
- `outputs/cluster_size_comparison.png`
## 7. Qualitative Analysis
- Top terms per cluster
- Frequent patterns (`outputs/cluster_patterns.csv`)
- Interpretable cluster labels
## 8. Key Findings
- Best representation
- Best algorithm
- Stability observations
- Practical interpretation for query grouping
## 9. Limitations and Future Work
- Small dataset limitations
- Sparse lexical overlap
- Potential improvements (embeddings, DBSCAN/hierarchical variants)
## 10. Conclusion
- Short final takeaway in 4-6 lines
## 11. Reproducibility Appendix
Commands used:
```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py
```
```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py --finalize
```
Main consolidated artifact for report extraction:
- `outputs/final_submission_metrics.csv`
