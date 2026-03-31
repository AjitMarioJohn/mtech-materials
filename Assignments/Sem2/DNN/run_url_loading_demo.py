from __future__ import annotations

from src.assignment_results import get_assignment_results, load_assignment_results_from_url  # noqa: F401


def example_local_loading() -> None:
    """Example 1: Load results from local computation."""
    print("=" * 60)
    print("Example 1: Local Loading (Generate Results)")
    print("=" * 60)

    results = get_assignment_results()

    print(f"Dataset: {results['dataset_name']}")
    print(f"Problem type: {results['problem_type']}")
    print(f"\nBaseline test F1: {results['baseline_model']['test_metrics']['f1']:.4f}")
    print(f"MLP test F1:      {results['mlp_model']['test_metrics']['f1']:.4f}")
    print()


def example_url_loading() -> None:
    """Example 2: Load results from a URL."""
    print("=" * 60)
    print("Example 2: URL Loading (Remote JSON)")
    print("=" * 60)
    print("""
To use URL loading:

1. Generate results locally:
   python run_step4.py
   
2. Upload data/assignment_results.json to a server:
   - GitHub (use raw.githubusercontent.com)
   - Google Drive (share publicly)
   - Any HTTP(S) server

3. Load in Python:
   from src.assignment_results import load_assignment_results_from_url
   
   url = 'https://example.com/data/assignment_results.json'
   results = load_assignment_results_from_url(url)
   
   print(f"Baseline F1: {results['baseline_model']['test_metrics']['f1']:.4f}")
   print(f"MLP F1: {results['mlp_model']['test_metrics']['f1']:.4f}")

Example URLs:
- GitHub Raw:  https://raw.githubusercontent.com/user/repo/main/data/assignment_results.json
- Google Drive: https://drive.google.com/uc?export=download&id=FILE_ID
""")
    print()


def main() -> None:
    example_local_loading()
    example_url_loading()


if __name__ == "__main__":
    main()

