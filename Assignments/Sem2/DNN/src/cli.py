from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.assignment_results import get_assignment_results, load_assignment_results_from_url
from src.data_pipeline import load_adult_income_data, preview_adult_preprocessing
from src.logistic_regression import LogisticRegressionScratch
from src.metrics import binary_classification_metrics
from src.mlp import MLPScratch

DATA_DIR = Path("data")
LOSS_CURVES_PATH = DATA_DIR / "loss_curves.png"
PERF_COMPARISON_PATH = DATA_DIR / "performance_comparison.png"
CONFUSION_MATRICES_PATH = DATA_DIR / "confusion_matrices.png"
RESULTS_JSON_PATH = DATA_DIR / "assignment_results.json"


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")
    if {"tp", "tn", "fp", "fn"}.issubset(metrics.keys()):
        print(
            "  Confusion: "
            f"TP={int(metrics['tp'])}, TN={int(metrics['tn'])}, "
            f"FP={int(metrics['fp'])}, FN={int(metrics['fn'])}"
        )


def run_step1() -> None:
    data = load_adult_income_data(test_size=0.2, random_state=42)

    print(f"Dataset: {data.dataset_name}")
    print(f"Original shape: samples={data.n_samples}, features={data.n_features}")
    print(f"Train shape: X={data.X_train.shape}, y={data.y_train.shape}")
    print(f"Test shape: X={data.X_test.shape}, y={data.y_test.shape}")
    print(f"Encoded feature count: {len(data.feature_names)}")
    print(f"Train positive class ratio: {data.y_train.mean():.4f}")
    print(f"Test positive class ratio: {data.y_test.mean():.4f}")


def run_step2() -> None:
    data = load_adult_income_data(test_size=0.2, random_state=42)

    model = LogisticRegressionScratch(learning_rate=0.1, epochs=800)
    start = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    train_time_seconds = time.perf_counter() - start

    y_train_pred = model.predict(data.X_train)
    y_test_pred = model.predict(data.X_test)

    train_metrics = binary_classification_metrics(data.y_train, y_train_pred)
    test_metrics = binary_classification_metrics(data.y_test, y_test_pred)

    print(f"Dataset: {data.dataset_name}")
    print(f"Train samples: {data.X_train.shape[0]}, Test samples: {data.X_test.shape[0]}")
    print(f"Feature count after preprocessing: {data.X_train.shape[1]}")
    print(f"Training time (seconds): {train_time_seconds:.3f}")
    print(f"Initial loss: {model.loss_history[0]:.6f}")
    print(f"Final loss  : {model.loss_history[-1]:.6f}")

    _print_metrics("Training metrics", train_metrics)
    _print_metrics("Test metrics", test_metrics)


def run_step3() -> None:
    data = load_adult_income_data(test_size=0.2, random_state=42)

    architecture = [data.X_train.shape[1], 128, 64, 1]
    model = MLPScratch(
        architecture=architecture,
        learning_rate=0.01,
        epochs=700,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    train_time_seconds = time.perf_counter() - start

    y_train_pred = model.predict(data.X_train)
    y_test_pred = model.predict(data.X_test)

    train_metrics = binary_classification_metrics(data.y_train, y_train_pred)
    test_metrics = binary_classification_metrics(data.y_test, y_test_pred)

    print(f"Dataset: {data.dataset_name}")
    print(f"MLP architecture: {architecture}")
    print(f"Train samples: {data.X_train.shape[0]}, Test samples: {data.X_test.shape[0]}")
    print(f"Training time (seconds): {train_time_seconds:.3f}")
    print(f"Initial loss: {model.loss_history[0]:.6f}")
    print(f"Final loss  : {model.loss_history[-1]:.6f}")

    _print_metrics("Training metrics", train_metrics)
    _print_metrics("Test metrics", test_metrics)


def _plot_loss_curves(baseline_loss: list[float], mlp_loss: list[float]) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.plot(baseline_loss, label="Baseline (LogReg)")
    plt.plot(mlp_loss, label="MLP")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVES_PATH, dpi=150)
    plt.close()


def _plot_performance_comparison(baseline_test: dict[str, float], mlp_test: dict[str, float]) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    baseline_values = [baseline_test[m] for m in metrics]
    mlp_values = [mlp_test[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width / 2, baseline_values, width=width, label="Baseline")
    plt.bar(x + width / 2, mlp_values, width=width, label="MLP")
    plt.xticks(x, [m.upper() for m in metrics])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Test Performance Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PERF_COMPARISON_PATH, dpi=150)
    plt.close()


def _plot_confusion_matrices(baseline_test: dict[str, float], mlp_test: dict[str, float]) -> None:
    baseline_cm = np.array(
        [
            [int(baseline_test["tn"]), int(baseline_test["fp"])],
            [int(baseline_test["fn"]), int(baseline_test["tp"])],
        ]
    )
    mlp_cm = np.array(
        [
            [int(mlp_test["tn"]), int(mlp_test["fp"])],
            [int(mlp_test["fn"]), int(mlp_test["tp"])],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, cm, title in zip(
        axes,
        [baseline_cm, mlp_cm],
        ["Baseline Confusion Matrix", "MLP Confusion Matrix"],
    ):
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["<=50K", ">50K"])
        ax.set_yticks([0, 1], ["<=50K", ">50K"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRICES_PATH, dpi=150)
    plt.close()


def run_step4() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = get_assignment_results()
    baseline = results["baseline_model"]
    mlp = results["mlp_model"]

    _plot_loss_curves(
        baseline_loss=baseline["loss_history"],
        mlp_loss=mlp["loss_history"],
    )
    _plot_performance_comparison(
        baseline_test=baseline["test_metrics"],
        mlp_test=mlp["test_metrics"],
    )
    _plot_confusion_matrices(
        baseline_test=baseline["test_metrics"],
        mlp_test=mlp["test_metrics"],
    )

    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Dataset: {results['dataset_name']}")
    print(f"Samples: {results['n_samples']}, Raw features: {results['n_features']}")
    print(f"Problem type: {results['problem_type']}, Primary metric: {results['primary_metric']}")

    print("\nBaseline test metrics:")
    for metric in ("accuracy", "precision", "recall", "f1"):
        print(f"  {metric}: {baseline['test_metrics'][metric]:.4f}")
    print(f"  training_time_seconds: {baseline['training_time_seconds']:.3f}")

    print("\nMLP test metrics:")
    for metric in ("accuracy", "precision", "recall", "f1"):
        print(f"  {metric}: {mlp['test_metrics'][metric]:.4f}")
    print(f"  architecture: {mlp['architecture']}")
    print(f"  training_time_seconds: {mlp['training_time_seconds']:.3f}")

    print(f"\nSaved: {LOSS_CURVES_PATH}")
    print(f"Saved: {PERF_COMPARISON_PATH}")
    print(f"Saved: {CONFUSION_MATRICES_PATH}")
    print(f"Saved: {RESULTS_JSON_PATH}")


def run_url_demo() -> None:
    print("=" * 60)
    print("Example 1: Local Loading (Generate Results)")
    print("=" * 60)
    results = get_assignment_results()
    print(f"Dataset: {results['dataset_name']}")
    print(f"Problem type: {results['problem_type']}")
    print(f"\nBaseline test F1: {results['baseline_model']['test_metrics']['f1']:.4f}")
    print(f"MLP test F1:      {results['mlp_model']['test_metrics']['f1']:.4f}")

    print("\n" + "=" * 60)
    print("Example 2: URL Loading (Remote JSON)")
    print("=" * 60)
    print(
        "Pass --results-url to fetch JSON from a hosted endpoint. "
        "Example: https://raw.githubusercontent.com/user/repo/main/data/assignment_results.json"
    )


def run_preview(rows: int) -> None:
    before_df, after_df = preview_adult_preprocessing(n_rows=rows)
    print("Before preprocessing:")
    print(before_df)
    print("\nAfter preprocessing (first 20 columns):")
    print(after_df.iloc[:, :20])
    print(f"\nTransformed shape for preview rows: {after_df.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DNN assignment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("step1", help="Run milestone 1 data pipeline")
    subparsers.add_parser("step2", help="Run baseline logistic regression")
    subparsers.add_parser("step3", help="Run MLP training and evaluation")
    subparsers.add_parser("step4", help="Generate final metrics and plots")

    preview_parser = subparsers.add_parser("preview", help="Show before/after preprocessing samples")
    preview_parser.add_argument("--rows", type=int, default=5, help="Number of rows to preview")

    url_parser = subparsers.add_parser("url", help="Load assignment results from a URL")
    url_parser.add_argument("--results-url", required=True, help="URL hosting assignment_results.json")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "step1":
        run_step1()
    elif args.command == "step2":
        run_step2()
    elif args.command == "step3":
        run_step3()
    elif args.command == "step4":
        run_step4()
    elif args.command == "preview":
        run_preview(rows=args.rows)
    elif args.command == "url":
        results = load_assignment_results_from_url(args.results_url)
        print(f"Dataset: {results.get('dataset_name', 'unknown')}")
        baseline = results.get("baseline_model", {}).get("test_metrics", {})
        mlp = results.get("mlp_model", {}).get("test_metrics", {})
        if baseline:
            print(f"Baseline F1: {baseline.get('f1', float('nan')):.4f}")
        if mlp:
            print(f"MLP F1: {mlp.get('f1', float('nan')):.4f}")


if __name__ == "__main__":
    main()

