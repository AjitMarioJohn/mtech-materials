# src/patterns.py
import re
from collections import Counter
from typing import Iterable


def _make_ngrams(tokens: list[str], n: int) -> Iterable[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i+n])


def extract_top_ngrams(texts: list[str], ngram_range=(1, 2), top_n: int = 10) -> list[tuple[str, int]]:
    counter = Counter()
    min_n, max_n = ngram_range

    for text in texts:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for n in range(min_n, max_n + 1):
            counter.update(_make_ngrams(tokens, n))

    return counter.most_common(top_n)


def extract_cluster_patterns(queries: list[str], labels, ngram_range=(1, 2), top_n: int = 10) -> dict[int, list[tuple[str, int]]]:
    clusters: dict[int, list[str]] = {}
    for q, lbl in zip(queries, labels):
        cid = int(lbl)
        clusters.setdefault(cid, []).append(q)

    return {
        cid: extract_top_ngrams(texts, ngram_range=ngram_range, top_n=top_n)
        for cid, texts in sorted(clusters.items())
    }
