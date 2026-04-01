from __future__ import annotations


def _is_good_label_token(token: str) -> bool:
    token = token.strip().lower()
    if not token:
        return False
    if token.isdigit():
        return False
    return True


def build_cluster_labels(
    top_terms: dict[int, list[str]],
    cluster_patterns: dict[int, list[tuple[str, int]]] | None = None,
    max_terms: int = 2,
) -> dict[int, str]:
    """
    Build short human-readable labels per cluster from top terms and optional patterns.
    Example label: 'python + tutorial'
    """
    labels: dict[int, str] = {}
    cluster_patterns = cluster_patterns or {}

    for cid in sorted(top_terms):
        chosen: list[str] = []

        # Prefer top terms first
        for term in top_terms.get(cid, []):
            if _is_good_label_token(term):
                chosen.append(term)
            if len(chosen) >= max_terms:
                break

        # Backfill from frequent patterns if needed
        if len(chosen) < max_terms:
            for pattern, _count in cluster_patterns.get(cid, []):
                if _is_good_label_token(pattern) and pattern not in chosen:
                    chosen.append(pattern)
                if len(chosen) >= max_terms:
                    break

        if not chosen:
            labels[cid] = f"cluster_{cid}"
        else:
            labels[cid] = " + ".join(chosen[:max_terms])

    return labels


def labels_to_rows(
    labels: dict[int, str],
    top_terms: dict[int, list[str]],
) -> list[dict]:
    rows: list[dict] = []
    for cid in sorted(labels):
        rows.append(
            {
                "cluster_id": int(cid),
                "label": labels[cid],
                "top_terms": ", ".join(top_terms.get(cid, [])),
            }
        )
    return rows
