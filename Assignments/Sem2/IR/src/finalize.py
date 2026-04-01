from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.read_csv(str(path)))


def build_final_submission_metrics(
    outputs_dir: str | Path = "outputs",
    output_file: str | Path = "outputs/final_submission_metrics.csv",
) -> Path:
    """
    Build one consolidated long-format metrics CSV from generated outputs.
    """
    outputs_path = Path(outputs_dir)
    rows: list[dict] = []

    # 1) Representation experiments
    exp_df = _safe_read_csv(outputs_path / "experiment_summary.csv")
    if not exp_df.empty:
        for _, row in exp_df.iterrows():
            rows.append(
                {
                    "section": "representation_sweep",
                    "source_file": "experiment_summary.csv",
                    "representation": row.get("representation", ""),
                    "algorithm": "kmeans",
                    "ngram_range": row.get("ngram_range", ""),
                    "n_clusters": row.get("n_clusters", ""),
                    "seed": "",
                    "metric_name": "silhouette",
                    "metric_value": row.get("silhouette", ""),
                    "extra": row.get("cluster_sizes", ""),
                }
            )

    best_rep_df = _safe_read_csv(outputs_path / "representation_best_summary.csv")
    if not best_rep_df.empty:
        for _, row in best_rep_df.iterrows():
            rows.append(
                {
                    "section": "representation_best",
                    "source_file": "representation_best_summary.csv",
                    "representation": row.get("representation", ""),
                    "algorithm": "kmeans",
                    "ngram_range": row.get("ngram_range", ""),
                    "n_clusters": row.get("n_clusters", ""),
                    "seed": "",
                    "metric_name": "silhouette",
                    "metric_value": row.get("silhouette", ""),
                    "extra": row.get("cluster_sizes", ""),
                }
            )

    # 2) Algorithm comparison
    algo_df = _safe_read_csv(outputs_path / "algorithm_comparison.csv")
    if not algo_df.empty:
        for _, row in algo_df.iterrows():
            rows.append(
                {
                    "section": "algorithm_comparison",
                    "source_file": "algorithm_comparison.csv",
                    "representation": row.get("representation", ""),
                    "algorithm": row.get("algorithm", ""),
                    "ngram_range": row.get("ngram_range", ""),
                    "n_clusters": row.get("n_clusters", ""),
                    "seed": "",
                    "metric_name": "silhouette",
                    "metric_value": row.get("silhouette", ""),
                    "extra": row.get("cluster_sizes", ""),
                }
            )

    labels_df = _safe_read_csv(outputs_path / "cluster_labels_comparison.csv")
    if not labels_df.empty:
        for _, row in labels_df.iterrows():
            rows.append(
                {
                    "section": "cluster_labels",
                    "source_file": "cluster_labels_comparison.csv",
                    "representation": "",
                    "algorithm": row.get("algorithm", ""),
                    "ngram_range": "",
                    "n_clusters": "",
                    "seed": "",
                    "metric_name": "cluster_label",
                    "metric_value": row.get("label", ""),
                    "extra": f"cluster_id={row.get('cluster_id', '')}; top_terms={row.get('top_terms', '')}",
                }
            )

    # 3) Stability
    stab_df = _safe_read_csv(outputs_path / "seed_stability_summary.csv")
    if not stab_df.empty:
        for _, row in stab_df.iterrows():
            rows.append(
                {
                    "section": "seed_stability",
                    "source_file": "seed_stability_summary.csv",
                    "representation": "tfidf_unigram",
                    "algorithm": "kmeans",
                    "ngram_range": row.get("ngram_range", ""),
                    "n_clusters": row.get("n_clusters", ""),
                    "seed": row.get("seed", ""),
                    "metric_name": "silhouette",
                    "metric_value": row.get("silhouette", ""),
                    "extra": row.get("cluster_sizes", ""),
                }
            )
            rows.append(
                {
                    "section": "seed_stability",
                    "source_file": "seed_stability_summary.csv",
                    "representation": "tfidf_unigram",
                    "algorithm": "kmeans",
                    "ngram_range": row.get("ngram_range", ""),
                    "n_clusters": row.get("n_clusters", ""),
                    "seed": row.get("seed", ""),
                    "metric_name": "ari_vs_seed_1",
                    "metric_value": row.get("ari_vs_seed_1", ""),
                    "extra": row.get("cluster_sizes", ""),
                }
            )

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

