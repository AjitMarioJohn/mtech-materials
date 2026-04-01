from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_cluster_sizes(text: str) -> dict[int, int]:
    sizes: dict[int, int] = {}
    if pd.isna(text):
        return sizes
    for part in str(text).split("|"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        cluster_id, size = part.split(":", 1)
        try:
            sizes[int(cluster_id.strip())] = int(size.strip())
        except ValueError:
            continue
    return sizes


def generate_report_plots(
    algorithm_comparison_csv: str | Path = "outputs/algorithm_comparison.csv",
    output_dir: str | Path = "outputs",
) -> dict[str, Path]:
    """
    Generate simple report-ready plots from existing experiment outputs.
    Returns the generated file paths.
    """
    output_path = _ensure_output_dir(output_dir)
    comparison_path = Path(algorithm_comparison_csv)
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {comparison_path}")

    df = pd.read_csv(str(comparison_path))
    required_columns = {"algorithm", "silhouette", "cluster_sizes"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {comparison_path}: {sorted(missing)}")

    generated: dict[str, Path] = {}

    # 1) Silhouette comparison bar chart
    silhouette_file = output_path / "silhouette_comparison.png"
    plt.figure(figsize=(8, 4.5))
    plot_df = df.copy()
    plot_df["silhouette"] = pd.to_numeric(plot_df["silhouette"], errors="coerce")
    plot_df = plot_df.dropna(subset=["silhouette"])
    if not plot_df.empty:
        plt.bar(plot_df["algorithm"], plot_df["silhouette"], color=["#4C78A8", "#F58518"][: len(plot_df)])
        plt.title("Silhouette Score Comparison")
        plt.ylabel("Silhouette score")
        plt.ylim(0, max(0.1, float(plot_df["silhouette"].max()) * 1.25))
        for idx, value in enumerate(plot_df["silhouette"].tolist()):
            plt.text(idx, value + 0.002, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    else:
        plt.text(0.5, 0.5, "No silhouette data available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(silhouette_file, dpi=200)
    plt.close()
    generated["silhouette_comparison"] = silhouette_file

    # 2) Cluster size comparison chart
    cluster_file = output_path / "cluster_size_comparison.png"
    parsed_rows = []
    for _, row in df.iterrows():
        sizes = _parse_cluster_sizes(row["cluster_sizes"])
        for cluster_id, size in sizes.items():
            parsed_rows.append(
                {
                    "algorithm": row["algorithm"],
                    "cluster_id": cluster_id,
                    "size": size,
                }
            )

    sizes_df = pd.DataFrame(parsed_rows)
    plt.figure(figsize=(9, 4.8))
    if not sizes_df.empty:
        pivot = sizes_df.pivot_table(index="cluster_id", columns="algorithm", values="size", fill_value=0)
        pivot.plot(kind="bar", ax=plt.gca(), width=0.8)
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of queries")
        plt.title("Cluster Size Distribution by Algorithm")
        plt.legend(title="Algorithm")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No cluster size data available", ha="center", va="center")
        plt.axis("off")
    plt.savefig(cluster_file, dpi=200)
    plt.close()
    generated["cluster_size_comparison"] = cluster_file

    return generated

