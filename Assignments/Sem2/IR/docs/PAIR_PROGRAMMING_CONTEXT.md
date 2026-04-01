# IR Assignment Pair-Programming Context

Last updated: 2026-04-03 (Session 1 closed, Session 2 ready)
Workspace: `C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR`

## Assignment
- Course: Information Retrieval (S2-25_AIMLZG537)
- Assignment: Query Clustering for Intent Discovery in Web Search

## What we completed so far
1. Extracted assignment requirements from PDF.
2. Ran diagnostic to assess current level.
3. Defined personalized roadmap and peer-programming workflow.
4. Environment setup was completed by user.
5. Decision: start with a small local dataset first, then switch to a public dataset later.

## Diagnostic snapshot
- Strengths:
  - Solid software engineering background (Java, 10 years).
  - Basic TF-IDF intuition.
- Gaps to focus first:
  - IR evaluation metrics: precision/recall/F1, MAP/MRR/nDCG.
  - Clustering interpretation and stability analysis.
  - Experiment design and error analysis depth.
  - Python ML workflow confidence (self-rated 2/5).

## Extracted assignment rules (must satisfy)
- Implement at least 2 query representations (from TF-IDF, n-grams, embeddings).
- Implement at least 2 clustering algorithms (from K-means, hierarchical, DBSCAN).
- Perform hyperparameter experiments.
- Analyze cluster stability.
- Include text mining component:
  - frequent intent pattern extraction,
  - cluster labeling using top terms.
- Evaluate using silhouette score.
- Demonstrate improvement in query suggestions or retrieval grouping.
- Deliverables:
  - ZIP with source code + PDF report (with visualizations),
  - PDF naming: `Group#-<No>-Assignment-1.pdf`,
  - assumptions clearly documented,
  - if ZIP > 10MB, include source-code link in PDF.

## Planned implementation roadmap
1. Session 1: scaffold project + tiny dataset + TF-IDF + KMeans baseline.
2. Session 2: add second representation (n-gram), compare quality.
3. Session 3: add second clustering (DBSCAN or hierarchical).
4. Session 4: hyperparameter sweeps + stability checks.
5. Session 5: cluster labels + frequent patterns + silhouette + qualitative analysis.
6. Session 6: visualizations + report packaging.

## Current status
- Environment setup: Done.
- Dataset: done (`data/queries.csv` with 20 starter queries).
- Code scaffold: done (baseline modules + runner created).
- Baseline pipeline: done (TF-IDF + KMeans + CSV export + silhouette + top terms).
- Session 1 closure items: done
  - Added controlled experiment mode in `main.py` for:
    - `ngram_range`: `(1,1)` vs `(1,2)`
    - `n_clusters`: `3, 4, 5`
  - Added experiment export: `outputs/experiment_summary.csv`
  - Updated reproducible setup docs in `docs/README.md`
  - Updated dependencies in `requirements.txt`

## Session log
- 2026-04-01 (Session 1 kickoff):
  - User is ready to start implementation with guided pair-programming.
  - Agreed style: explain-first, user codes, assistant reviews and iterates.
  - Immediate focus: tiny local dataset + TF-IDF + KMeans smoke pipeline.
- 2026-04-01 (Session 1 implementation):
  - Implemented loader in `src/queries.py`: `load_queries(csv_path)` with validation.
  - Implemented TF-IDF in `src/features.py`: `build_tfidf_features(...)`.
  - Implemented clustering in `src/cluster.py`: `run_kmeans_baseline(...)`.
  - Added evaluation + interpretation in `src/cluster.py`:
    - `compute_silhouette_score(X, labels)`
    - `get_top_terms_per_cluster(model, vectorizer, top_n=5)`
  - Built end-to-end runner in `main.py` and exported `outputs/cluster_assignments.csv`.
  - Fixed bug caused by using `_` for both model and vectorizer in `main.py`.
  - Latest run output:
    - Silhouette score: `0.0744`
    - Cluster sizes: `0:2, 1:3, 2:11, 3:4`
    - Top terms were printed successfully for all clusters.
  - User requested explanation-first mode because code felt complex; switched to step-by-step teaching.

## Next immediate task when resuming
- Start Session 2 goal: second representation/experiment track.
- Add Representation #2 track (choose one):
  - n-gram TF-IDF as explicit second representation, or
  - embedding-based representation (if feasible in assignment scope).
- Add second clustering algorithm:
  - hierarchical (recommended for small dataset explainability) or DBSCAN.
- Compare algorithms on shared metrics:
  - silhouette,
  - cluster size distribution,
  - qualitative top-term label quality.

## Resume prompt template
Use this exact text in next session:

"Continue from `PAIR_PROGRAMMING_CONTEXT.md`. Session 1 baseline is done (TF-IDF + KMeans + silhouette + top terms + CSV export). Please continue in explain-first pair-programming mode. Start by explaining existing code simply, then guide me through n-gram and k sweep experiments step-by-step (I write code, you review)."

## Session 2 kickoff checklist
- Review `outputs/experiment_summary.csv` and pick best baseline config.
- Implement second clustering algorithm in `src/cluster.py` or a new module.
- Add comparison run mode in `main.py` for baseline vs second algorithm.
- Export per-algorithm comparison table to `outputs/algorithm_comparison.csv`.
- Add visuals in report draft (silhouette bar chart + cluster size chart).

