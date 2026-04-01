# ZIP-Ready Manifest

Use this file as the final check before creating the submission ZIP.

## Include in the ZIP

### Notebook / report
- `IR_Large_Dataset_Report.ipynb`
- Exported PDF version of the notebook

### Code
- `main.py`
- `requirements.txt`
- `src/queries.py`
- `src/features.py`
- `src/cluster.py`
- `src/data_loader.py`
- `src/patterns.py`
- `src/labels.py`
- `src/report_plots.py`
- `src/finalize.py`

### Data
- `data/queries.csv`
- `data/queries_large.csv`

### Output artifacts
- `outputs/experiment_summary.csv`
- `outputs/representation_best_summary.csv`
- `outputs/algorithm_comparison.csv`
- `outputs/cluster_labels_comparison.csv`
- `outputs/seed_stability_summary.csv`
- `outputs/final_submission_metrics.csv`
- `outputs/silhouette_comparison.png`
- `outputs/cluster_size_comparison.png`
- `outputs/cluster_patterns.csv`
- `outputs/cluster_assignments.csv`
- `outputs/cluster_assignments_large.csv`

### Documentation
- `docs/REPORT_ANALYSIS.md`
- `docs/REPORT_TABLES.md`
- `docs/FINAL_REPORT_STRUCTURE.md`
- `docs/SUBMISSION_CHECKLIST.md`
- `docs/ZIP_READY_MANIFEST.md`

## Final checks before zipping
- [ ] Notebook runs end-to-end without errors.
- [ ] Notebook exports to PDF correctly.
- [ ] Report tables match CSV outputs.
- [ ] Figures are visible and readable.
- [ ] ZIP contains the report PDF at the root or in a clear folder.
- [ ] If ZIP exceeds 10 MB, add a source-code link in the PDF.

## Recommended ZIP name
Use a clear name such as:

```text
Group#-<No>-Assignment-1.zip
```

## Minimal submission flow
1. Run the notebook.
2. Export the notebook to PDF.
3. Copy the files listed above into a submission folder.
4. Create the ZIP.
5. Re-open the ZIP and verify contents.

