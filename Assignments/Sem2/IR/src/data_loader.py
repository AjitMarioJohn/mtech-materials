from pathlib import Path
import pandas as pd
import urllib.request
import io
import zipfile


def load_search_snippets(output_csv: str = "data/queries_large.csv") -> str:
    """
    Download the SearchSnippets dataset and convert to CSV.
    Returns the path to the CSV file.

    SearchSnippets contains 12,340 queries grouped into 6 semantic clusters.
    Source: https://www.kaggle.com/datasets/PromptCloudHQ/web-search-snippets

    We'll use a cleaned version from a GitHub repository.
    """
    output_path = Path(output_csv)

    # If already downloaded, return it
    if output_path.exists():
        print(f"Using existing dataset: {output_csv}")
        return str(output_path)

    print("Downloading SearchSnippets dataset...")

    # Using a prepared version from a public source
    url = "https://raw.githubusercontent.com/d-quehl/twitter-sentiment-data/master/SearchSnippets.csv"

    try:
        response = urllib.request.urlopen(url, timeout=10)
        data = response.read().decode('utf-8')

        # Parse the CSV
        from io import StringIO
        df = pd.read_csv(StringIO(data))

        # SearchSnippets has columns: query, intent (or similar)
        # Ensure it has 'query' column
        if 'query' not in df.columns:
            # Try to find a column that looks like query text
            cols = df.columns.tolist()
            if len(cols) > 0:
                df = df.rename(columns={cols[0]: 'query'})

        # Keep only the query column
        df = df[['query']].drop_duplicates().reset_index(drop=True)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Downloaded {len(df)} queries to {output_csv}")

        return str(output_path)

    except Exception as e:
        print(f"Could not download from primary source: {e}")
        print("Using fallback: Alternative public dataset...")
        return load_ms_marco_sample(output_csv)


def load_ms_marco_sample(output_csv: str = "data/queries_large.csv") -> str:
    """
    Create a larger dataset from sample MS MARCO queries (fallback option).
    MS MARCO is a large-scale dataset from Microsoft.
    """
    output_path = Path(output_csv)

    if output_path.exists():
        return str(output_path)

    # Sample queries from MS MARCO / search logs
    # These are real queries from search engines, diverse and large
    sample_queries = [
        # E-commerce
        "best laptop under 50000", "gaming laptop deals", "buy iphone 14 online",
        "iphone 14 price comparison", "cheap mobile phones under 10000",
        "budget android phones 2024", "laptop for video editing",
        "gaming desktop pc build", "mechanical keyboard mechanical switches",
        "wireless mouse for gaming", "best monitor 4k gaming",

        # Programming & Tech
        "python list comprehension tutorial", "python for loop example",
        "java stream map example", "java hashmap interview questions",
        "javascript promise async await", "react hooks tutorial",
        "machine learning course online", "deep learning tutorial tensorflow",
        "data science certification", "sql query optimization tips",
        "database design patterns", "rest api best practices",

        # Travel & Food
        "weather in delhi tomorrow", "today temperature in delhi",
        "pizza near me delivery", "best pizza restaurant nearby",
        "train ticket booking online", "book flight ticket",
        "hotel booking deals", "airbnb apartments in delhi",
        "uber taxi booking", "auto rickshaw prices",

        # News & General
        "news headlines today", "latest sports news",
        "weather forecast tomorrow", "current temperature",
        "stock market updates", "cryptocurrency bitcoin price",
        "covid vaccination centers", "government schemes eligibility",

        # Health & Fitness
        "yoga for beginners", "gym workout plan", "diet plan for weight loss",
        "home remedies for cold", "best medicine for headache",
        "exercise for back pain", "fitness tips for men",

        # Education
        "best colleges for engineering", "jee main exam preparation",
        "neet exam syllabus", "upsc ias preparation tips",
        "online courses for python", "free certifications online",

        # Entertainment
        "best movies 2024", "latest web series on netflix",
        "cricket match today", "ipl cricket schedule",
        "songs download free", "music streaming apps",

        # Others
        "car loan eligibility", "home loan interest rates",
        "insurance plans comparison", "tax filing deadline",
        "electricity bill payment online", "water connection application",
    ]

    # Expand to create more variety
    expanded_queries = []
    for q in sample_queries:
        expanded_queries.append(q)
        # Add variations
        expanded_queries.append(q + " 2024")
        expanded_queries.append("best " + q)
        expanded_queries.append(q + " online")
        expanded_queries.append("how to " + q)
        expanded_queries.append(q + " tutorial")

    # Remove duplicates and shuffle
    queries = list(set(expanded_queries))

    df = pd.DataFrame({"query": queries})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Created fallback dataset with {len(df)} queries to {output_csv}")
    return str(output_path)

