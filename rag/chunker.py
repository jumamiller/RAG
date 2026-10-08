"""Stage 2: split long text into overlapping chunks.

Chunking is the #1 lever on retrieval quality. Chunks that are too big bury the
relevant sentence in noise; too small and they lose surrounding context. We use
a simple word-window with overlap so ideas spanning a boundary still appear
(mostly) intact in at least one chunk.

`chunk_size` and `overlap` are measured in words (a cheap proxy for tokens).
"""


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split `text` into overlapping word windows.

    Args:
        text: The raw text to split.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of chunk strings (empty list for empty input).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break  # last window already reached the end
    return chunks


if __name__ == "__main__":
    sample = " ".join(f"word{i}" for i in range(450))
    pieces = chunk_text(sample, chunk_size=200, overlap=40)
    print(f"{len(pieces)} chunks from 450 words:")
    for i, piece in enumerate(pieces):
        print(f"  chunk {i}: {len(piece.split())} words")
