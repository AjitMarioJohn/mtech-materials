from pathlib import Path
import sys
from collections import Counter

import pandas as pd

from sklearn.metrics import adjusted_rand_score

from src.queries import load_queries
from src.data_loader import load_search_snippets
from src.features import build_tfidf_features
from src.report_plots import generate_report_plots
from src.finalize import build_final_submission_metrics
from src.cluster import (
    run_kmeans_baseline,
    run_hierarchical_baseline,
    compute_silhouette_score,
    get_top_terms_per_cluster,
    get_top_terms_from_labels,
)
from src.labels import build_cluster_labels, labels_to_rows
from src.patterns import extract_cluster_patterns


def _cluster_sizes_as_text(labels) -> str:
    counts = Counter(int(x) for x in labels)
    return " | ".join(f"{cid}:{counts[cid]}" for cid in sorted(counts))


def run_single_pipeline(
    queries: list[str],
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    n_clusters: int,
    random_state: int,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"Building TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
    X, vectorizer = build_tfidf_features(
        queries,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
    )
    if verbose:
        print(f"Feature matrix shape: {X.shape}")

    if verbose:
        print(f"Running KMeans with {n_clusters} clusters...")
    labels, model = run_kmeans_baseline(X, n_clusters=n_clusters, random_state=random_state)

    sil = compute_silhouette_score(X, labels)
    top_terms = get_top_terms_per_cluster(model=model, vectorizer=vectorizer, top_n=5)
    cluster_sizes = _cluster_sizes_as_text(labels)
    cluster_patterns = extract_cluster_patterns(queries, labels, ngram_range=(1, 2), top_n=10)

    if output_path is not None:
        result_df = pd.DataFrame(
            {
                "query": queries,
                "cluster": [int(x) for x in labels],
            }
        )
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_file, index=False)
        if verbose:
            print(f"Saved: {output_file}")
            print(result_df.groupby("cluster").size().sort_index())
    return {
        "silhouette": sil,
        "top_terms": top_terms,
        "cluster_sizes": cluster_sizes,
        "cluster_patterns": cluster_patterns,
    }


def run_hierarchical_pipeline(
    queries: list[str],
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    n_clusters: int,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"Building TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
    X, vectorizer = build_tfidf_features(
        queries,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
    )
    if verbose:
        print(f"Feature matrix shape: {X.shape}")

    if verbose:
        print(f"Running Hierarchical Clustering with {n_clusters} clusters...")
    labels, model = run_hierarchical_baseline(X, n_clusters=n_clusters)

    sil = compute_silhouette_score(X, labels)
    top_terms = get_top_terms_from_labels(X=X, labels=labels, vectorizer=vectorizer, top_n=5)
    cluster_sizes = _cluster_sizes_as_text(labels)
    cluster_patterns = extract_cluster_patterns(queries, labels, ngram_range=(1, 2), top_n=10)

    if output_path is not None:
        result_df = pd.DataFrame(
            {
                "query": queries,
                "cluster": [int(x) for x in labels],
            }
        )
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_file, index=False)
        if verbose:
            print(f"Saved: {output_file}")
            print(result_df.groupby("cluster").size().sort_index())

    return {
        "silhouette": sil,
        "top_terms": top_terms,
        "cluster_sizes": cluster_sizes,
        "cluster_patterns": cluster_patterns,
    }


def run_experiment_sweep(
    queries: list[str],
    *,
    max_features: int,
    random_state: int = 42,
):
    ngram_grid = [(1, 1), (1, 2)]
    cluster_grid = [3, 4, 5]
    rows: list[dict] = []


    print("\nRunning controlled experiments (ngram_range x n_clusters)...")
    for ngram_range in ngram_grid:
        for n_clusters in cluster_grid:
            result = run_single_pipeline(
                queries,
                max_features=max_features,
                ngram_range=ngram_range,
                n_clusters=n_clusters,
                random_state=random_state,
                output_path=None,
                verbose=False,
            )

            if ngram_range == (1, 1):
                representation = "tfidf_unigram"
            elif ngram_range == (1, 2):
                representation = "tfidf_unigram_bigram"
            else:
                representation = f"tfidf_{ngram_range[0]}_{ngram_range[1]}"

            row = {
                "representation": representation,
                "ngram_range": f"{ngram_range[0]},{ngram_range[1]}",
                "n_clusters": n_clusters,
                "silhouette": result["silhouette"],
                "cluster_sizes": result["cluster_sizes"],
            }
            rows.append(row)
            print(
                f"- ngram={row['ngram_range']}, k={n_clusters}, "
                f"silhouette={row['silhouette'] if row['silhouette'] is not None else 'NA'}, "
                f"sizes={row['cluster_sizes']}"
            )

    df = pd.DataFrame(rows)
    summary_path = Path("outputs/experiment_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"\nSaved experiment summary: {summary_path}")

    valid_df = df[df["silhouette"].notna()].copy()
    if not valid_df.empty:
        best_idx = valid_df.groupby("representation")["silhouette"].idxmax()
        best_by_rep = valid_df.loc[best_idx].sort_values(["representation", "silhouette"], ascending=[True, False])

        best_path = Path("outputs/representation_best_summary.csv")
        best_by_rep.to_csv(best_path, index=False)
        print(f"Saved best-per-representation summary: {best_path}")

    if not valid_df.empty:
        best = valid_df.sort_values("silhouette", ascending=False).iloc[0]
        print(
            "Best config: "
            f"ngram={best['ngram_range']}, "
            f"k={int(best['n_clusters'])}, "
            f"silhouette={float(best['silhouette']):.4f}"
        )


def run_algorithm_comparison(
    queries: list[str],
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    n_clusters: int,
):
    print("\nRunning algorithm comparison (KMeans vs Hierarchical)...")

    experiments = [
        (
            "kmeans",
            run_single_pipeline,
            {
                "max_features": max_features,
                "ngram_range": ngram_range,
                "n_clusters": n_clusters,
                "random_state": 42,
                "output_path": None,
                "verbose": False,
            },
        ),
        (
            "hierarchical",
            run_hierarchical_pipeline,
            {
                "max_features": max_features,
                "ngram_range": ngram_range,
                "n_clusters": n_clusters,
                "output_path": None,
                "verbose": False,
            },
        ),
    ]

    rows: list[dict] = []
    all_label_rows: list[dict] = []
    for algorithm_name, runner, kwargs in experiments:
        result = runner(queries, **kwargs)

        cluster_labels = build_cluster_labels(
            top_terms=result["top_terms"],
            cluster_patterns=result.get("cluster_patterns", {}),
            max_terms=2,
        )

        label_rows = labels_to_rows(cluster_labels, result["top_terms"])
        labels_df = pd.DataFrame(label_rows)
        labels_path = Path(f"outputs/cluster_labels_{algorithm_name}.csv")
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(labels_path, index=False)
        print(f"Saved cluster labels: {labels_path}")

        for row in label_rows:
            all_label_rows.append(
                {
                    "algorithm": algorithm_name,
                    "cluster_id": row["cluster_id"],
                    "label": row["label"],
                    "top_terms": row["top_terms"],
                }
            )

        rows.append(
            {
                "algorithm": algorithm_name,
                "representation": "tfidf_unigram" if ngram_range == (1, 1) else "tfidf_unigram_bigram",
                "ngram_range": f"{ngram_range[0]},{ngram_range[1]}",
                "n_clusters": n_clusters,
                "silhouette": result["silhouette"],
                "cluster_sizes": result["cluster_sizes"],
            }
        )
        print(
            f"- {algorithm_name}: silhouette={result['silhouette'] if result['silhouette'] is not None else 'NA'}, "
            f"sizes={result['cluster_sizes']}"
        )

    comparison_labels_df = pd.DataFrame(all_label_rows)
    comparison_labels_path = Path("outputs/cluster_labels_comparison.csv")
    comparison_labels_df.to_csv(comparison_labels_path, index=False)
    print(f"Saved merged cluster labels: {comparison_labels_path}")

    df = pd.DataFrame(rows)
    output_path = Path("outputs/algorithm_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved algorithm comparison: {output_path}")

    valid_df = df[df["silhouette"].notna()].copy()
    if not valid_df.empty:
        best = valid_df.sort_values("silhouette", ascending=False).iloc[0]
        print(
            "Best algorithm: "
            f"{best['algorithm']} (silhouette={float(best['silhouette']):.4f})"
        )


def run_seed_stability(
    queries: list[str],
    *,
    max_features: int,
    ngram_range: tuple[int, int] = (1, 1),
    n_clusters: int = 4,
    seeds: list[int] | None = None,
):
    seed_values = seeds if seeds is not None else [1, 7, 21, 42, 99]

    print("\nRunning seed stability analysis (KMeans)...")
    X, _ = build_tfidf_features(
        queries,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
    )

    reference_labels = None
    rows: list[dict] = []
    for seed in seed_values:
        labels, _ = run_kmeans_baseline(X, n_clusters=n_clusters, random_state=seed)
        sil = compute_silhouette_score(X, labels)
        if reference_labels is None:
            ari = 1.0
            reference_labels = labels
        else:
            ari = float(adjusted_rand_score(reference_labels, labels))

        rows.append(
            {
                "seed": seed,
                "ngram_range": f"{ngram_range[0]},{ngram_range[1]}",
                "n_clusters": n_clusters,
                "silhouette": sil,
                "ari_vs_seed_1": ari,
                "cluster_sizes": _cluster_sizes_as_text(labels),
            }
        )
        print(
            f"- seed={seed}, silhouette={sil if sil is not None else 'NA'}, ari={ari:.4f}, sizes={rows[-1]['cluster_sizes']}"
        )

    df = pd.DataFrame(rows)
    output_path = Path("outputs/seed_stability_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved seed stability summary: {output_path}")

    if not df.empty:
        print(f"Mean ARI vs seed 1: {float(df['ari_vs_seed_1'].mean()):.4f}")


def run_report_generation():
    print("\nGenerating report plots from existing CSV summaries...")
    generated = generate_report_plots()
    for name, path in generated.items():
        print(f"- {name}: {path}")


def run_finalize_generation():
    print("\nGenerating consolidated final submission metrics...")
    metrics_path = build_final_submission_metrics()
    print(f"- final_submission_metrics: {metrics_path}")


def main(
    use_large_dataset: bool = True,
    run_sweep: bool = True,
    run_compare: bool = True,
    run_report: bool = True,
    run_stability: bool = True,
    run_finalize: bool = False,
):
    has_data_workflow = run_sweep or run_compare or run_stability
    if run_report and not has_data_workflow and not run_finalize:
        run_report_generation()
        return

    if run_finalize and not has_data_workflow and not run_report:
        run_finalize_generation()
        return

    # Use small or large dataset
    if use_large_dataset:
        print("Loading large public dataset (MS MARCO sample)...")
        csv_path = load_search_snippets()
        output_path = "outputs/cluster_assignments_large.csv"
    else:
        csv_path = "data/queries.csv"
        output_path = "outputs/cluster_assignments.csv"
        print(f"Loading small dataset from {csv_path}...")

    queries = load_queries(csv_path)
    print(f"Loaded {len(queries)} queries")

    # Adjust parameters based on dataset size
    max_features = 5000 if not use_large_dataset else 10000
    n_clusters = 4 if not use_large_dataset else 10

    if run_sweep:
        run_experiment_sweep(queries, max_features=max_features, random_state=42)

    if run_compare:
        run_algorithm_comparison(
            queries,
            max_features=max_features,
            ngram_range=(1, 1),
            n_clusters=n_clusters,
        )

    if run_stability:
        run_seed_stability(
            queries,
            max_features=max_features,
            ngram_range=(1, 1),
            n_clusters=n_clusters,
        )

    if not has_data_workflow:
        result = run_single_pipeline(
            queries,
            max_features=max_features,
            ngram_range=(1, 1),
            n_clusters=n_clusters,
            random_state=42,
            output_path=output_path,
            verbose=True,
        )

        sil = result["silhouette"]
        if sil is None:
            print("Silhouette: not available (need at least 2 clusters).")
        else:
            print(f"Silhouette score: {sil:.4f}")

        print("\nTop terms per cluster:")
        for cid in sorted(result["top_terms"]):
            print(f"Cluster {cid}: {', '.join(result['top_terms'][cid])}")

        pattern_rows = []
        for cid, pairs in result["cluster_patterns"].items():
            for pattern, count in pairs:
                pattern_rows.append(
                    {"cluster_id": int(cid), "pattern": pattern, "count": int(count)}
                )

        patterns_df = pd.DataFrame(pattern_rows)
        patterns_path = Path("outputs/cluster_patterns.csv")
        patterns_path.parent.mkdir(parents=True, exist_ok=True)
        patterns_df.to_csv(patterns_path, index=False)
        print(f"Saved: {patterns_path}")
        print(patterns_df.head(10))

    if run_report:
        run_report_generation()

    if run_finalize:
        run_finalize_generation()



if __name__ == "__main__":
    use_small = "--small" in sys.argv or "-M" in sys.argv
    use_large = ("--large" in sys.argv or "-L" in sys.argv) or not use_small

    run_sweep = "--sweep" in sys.argv or "-S" in sys.argv
    run_compare = "--compare" in sys.argv or "-C" in sys.argv
    run_report = "--report" in sys.argv or "-R" in sys.argv
    run_stability = "--stability" in sys.argv or "-T" in sys.argv
    run_finalize = "--finalize" in sys.argv or "-F" in sys.argv

    if not (run_sweep or run_compare or run_report or run_stability or run_finalize):
        # Default run: large dataset + all comparison/report workflows.
        run_sweep = True
        run_compare = True
        run_stability = True
        run_report = True
        run_finalize = True

    main(
        use_large_dataset=use_large,
        run_sweep=run_sweep,
        run_compare=run_compare,
        run_report=run_report,
        run_stability=run_stability,
        run_finalize=run_finalize,
    )
