# IR Assignment 1 - Query Clustering

## Setup

```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python -m pip install -r requirements.txt
```

## Default Run (Large Dataset + All Comparisons)

Running without flags now executes the full comparison workflow on the large dataset:
- representation sweep
- algorithm comparison
- seed stability
- report plot generation

```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py
```

Use `--small` (or `-M`) to force the small dataset when needed.

## Run Baseline (TF-IDF + KMeans)

```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py
```

Outputs:
- `outputs/cluster_assignments.csv`

## Run with Larger Dataset

```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py --large
```

Outputs:
- `outputs/cluster_assignments_large.csv`

## Run Controlled Experiments (Session 1 closing step)

This runs 6 experiments:
- `ngram_range`: `(1,1)` and `(1,2)`
- `n_clusters`: `3`, `4`, `5`

```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py --sweep
```

Outputs:
- `outputs/experiment_summary.csv`

## Generate Report Plots

This reads the existing comparison CSV and creates two PNGs:
- `outputs/silhouette_comparison.png`
- `outputs/cluster_size_comparison.png`

```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py --report
```

Outputs:
- `outputs/silhouette_comparison.png`
- `outputs/cluster_size_comparison.png`

## Seed Stability Analysis

This reruns KMeans across multiple random seeds and measures stability with ARI.

```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py --stability
```

Outputs:
- `outputs/seed_stability_summary.csv`

## Final Submission Metrics (Consolidated)

Build one consolidated CSV for report extraction and submission QA.

```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py --finalize
```

Outputs:
- `outputs/final_submission_metrics.csv`

## Report Analysis Note

For a short report-ready write-up of the current results, see:
- `docs/REPORT_ANALYSIS.md`

For a table-based appendix, see:
- `docs/REPORT_TABLES.md`

For final report assembly and submission checks, see:
- `docs/FINAL_REPORT_STRUCTURE.md`
- `docs/SUBMISSION_CHECKLIST.md`

## Project Structure

- `main.py` - runner for baseline and sweep experiments
- `src/queries.py` - query loading and validation
- `src/features.py` - TF-IDF feature creation
- `src/cluster.py` - clustering, silhouette, top terms
- `src/report_plots.py` - report charts from saved CSV summaries
- `src/data_loader.py` - larger dataset loader/generator
- `docs/` - documentation and session context
- `docs/REPORT_ANALYSIS.md` - short analysis for the report
- `docs/REPORT_TABLES.md` - report-ready result tables
- `outputs/seed_stability_summary.csv` - stability summary across seeds
- `outputs/final_submission_metrics.csv` - consolidated metrics for final report

