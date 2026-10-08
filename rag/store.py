"""Stage 4: an in-memory FAISS vector store.

FAISS stores only vectors and returns integer row IDs on search, so we keep
parallel `texts` and `sources` lists to map those IDs back to readable content.

We use `IndexFlatIP` (exact, brute-force, inner-product). Because we L2-normalize
every vector before adding/searching, the inner product equals cosine similarity.
"""

import faiss
import numpy as np


class VectorStore:
    """A tiny wrapper over a FAISS inner-product index plus the original text."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts: list[str] = []
        self.sources: list[str] = []

    def add(
        self,
        texts: list[str],
        vectors: list[list[float]],
        sources: list[str] | None = None,
    ) -> None:
        """Add chunks and their embeddings to the index."""
        if not texts:
            return
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must be the same length")

        # FAISS requires float32; embeddings come back as float64 lists.
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)  # in-place: makes inner product == cosine
        self.index.add(arr)

        self.texts.extend(texts)
        self.sources.extend(sources if sources is not None else [""] * len(texts))

    def search(
        self, query_vector: list[float], k: int = 4
    ) -> list[tuple[str, str, float]]:
        """Return the top-k (text, source, score) for a query embedding."""
        k = min(k, len(self.texts))
        if k == 0:
            return []

        arr = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(arr)
        scores, ids = self.index.search(arr, k)

        results: list[tuple[str, str, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            results.append((self.texts[idx], self.sources[idx], float(score)))
        return results

    def __len__(self) -> int:
        return len(self.texts)


if __name__ == "__main__":
    from .embedder import embed

    sentences = [
        "The cat sat on the warm windowsill.",
        "A kitten napped in the sunny window.",
        "Quarterly tax filings are due next week.",
        "The invoice must be paid before the deadline.",
        "Photosynthesis converts sunlight into energy.",
    ]
    store = VectorStore(dim=1536)
    store.add(sentences, embed(sentences))

    query = "a small feline resting"
    for text, _source, score in store.search(embed([query])[0], k=3):
        print(f"{score:.3f}  {text}")
