from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering


def compute_silhouette_score(X, labels) -> float | None:
    """
    Compute silhouette score when at least 2 clusters are present.
    Returns None when score is not valid.
    """
    unique_labels = set(int(x) for x in labels)
    if len(unique_labels) < 2:
        return None
    return float(silhouette_score(X, labels))


def get_top_terms_per_cluster(model, vectorizer, top_n: int = 5) -> dict[int, list[str]]:
    """
    Extract top terms from each KMeans centroid.
    """
    feature_names = vectorizer.get_feature_names_out()
    top_terms: dict[int, list[str]] = {}

    # Ensure we don't request more top terms than features available
    n_features = len(feature_names)
    n_top = min(top_n, n_features)

    for cluster_id, center in enumerate(model.cluster_centers_):

        top_indices = center.argsort()[-n_top:][::-1]
        top_terms[cluster_id] = [feature_names[i] for i in top_indices]

    return top_terms


def get_top_terms_from_labels(X, labels, vectorizer, top_n: int = 5) -> dict[int, list[str]]:
    """
    Extract top terms per cluster using mean TF-IDF weights of cluster members.
    Works for algorithms that do not expose cluster_centers_ (e.g., hierarchical).
    """
    feature_names = vectorizer.get_feature_names_out()
    n_features = len(feature_names)
    n_top = min(top_n, n_features)

    top_terms: dict[int, list[str]] = {}
    unique_labels = sorted(set(int(x) for x in labels))

    for cluster_id in unique_labels:
        mask = [int(x) == cluster_id for x in labels]
        cluster_matrix = X[mask]

        # Mean TF-IDF weight per term for this cluster
        mean_vector = cluster_matrix.mean(axis=0)
        mean_scores = mean_vector.A1  # flatten sparse matrix row to 1D array

        top_indices = mean_scores.argsort()[-n_top:][::-1]
        top_terms[cluster_id] = [feature_names[i] for i in top_indices]

    return top_terms


def run_kmeans_baseline(X, n_clusters: int = 4, random_state: int = 42):
    """
    Run KMeans on TF-IDF features and return labels + fitted model.
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(X)
    return labels, model


def run_hierarchical_baseline(X, n_clusters: int = 4, linkage: str = "ward"):
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")

    X_dense = X.toarray()
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )
    labels = model.fit_predict(X_dense)
    return labels, model
