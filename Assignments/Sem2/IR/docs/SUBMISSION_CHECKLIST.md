# Submission Checklist (Assignment 1)
Use this checklist before creating the final ZIP and PDF.
## A. Required Deliverables
- [ ] Source code is complete and runs from clean setup.
- [ ] Final report PDF is prepared.
- [ ] PDF filename follows: `Group#-<No>-Assignment-1.pdf`.
- [ ] ZIP contains code + report + required outputs.
- [ ] If ZIP size exceeds 10 MB, include source-code link inside PDF.
## B. Implementation Requirements
- [ ] At least 2 query representations implemented.
- [ ] At least 2 clustering algorithms implemented.
- [ ] Hyperparameter experiments completed.
- [ ] Cluster stability analysis completed.
- [ ] Frequent intent pattern extraction included.
- [ ] Cluster labeling using top terms included.
- [ ] Silhouette score evaluation reported.
- [ ] Improvement in grouping/suggestions discussed.
## C. Must-Have Artifacts
- [ ] `outputs/experiment_summary.csv`
- [ ] `outputs/representation_best_summary.csv`
- [ ] `outputs/algorithm_comparison.csv`
- [ ] `outputs/cluster_labels_comparison.csv`
- [ ] `outputs/seed_stability_summary.csv`
- [ ] `outputs/final_submission_metrics.csv`
- [ ] `outputs/silhouette_comparison.png`
- [ ] `outputs/cluster_size_comparison.png`
## D. Recommended Docs in ZIP
- [ ] `docs/REPORT_ANALYSIS.md`
- [ ] `docs/REPORT_TABLES.md`
- [ ] `docs/FINAL_REPORT_STRUCTURE.md`
- [ ] `docs/SUBMISSION_CHECKLIST.md`
## E. Reproducibility Commands
Run these before final packaging:
```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python -m pip install -r requirements.txt
```
```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py
```
```powershell
cd "C:\Projects\AI-ML\mtech-materials\Assignments\Sem2\IR"
python main.py --finalize
```
## F. Final Manual Review
- [ ] Report numbers match values in output CSV files.
- [ ] All figure captions are clear and referenced in report text.
- [ ] Assumptions and limitations are explicitly stated.
- [ ] No placeholder text remains.
- [ ] Report PDF opens correctly and all pages render.
## G. Packaging
- [ ] Create final ZIP from project root.
- [ ] Include report PDF in ZIP root or clearly named folder.
- [ ] Re-open ZIP and verify file integrity before submission.
