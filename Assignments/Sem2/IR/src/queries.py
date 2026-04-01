from pathlib import Path
import pandas as pd


def load_queries(csv_path: str) -> list[str]:
    """
    Load queries from a CSV file with a required 'query' column.
    Returns a cleaned list of non-empty query strings.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError("CSV must contain a 'query' column.")

    queries = (
        df["query"]
        .astype(str)
        .str.strip()
    )
    queries = queries[queries != ""].tolist()

    if not queries:
        raise ValueError("No valid queries found after cleaning.")

    return queries
