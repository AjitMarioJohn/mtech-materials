from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(
    queries: list[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 1,
):
    """
    Convert query text into TF-IDF sparse matrix.
    Returns (matrix, vectorizer).
    """
    if not queries:
        raise ValueError("queries list is empty.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    X = vectorizer.fit_transform(queries)
    return X, vectorizer
