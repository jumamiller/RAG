"""Stage 1: load raw text from local documents.

Reads every supported file under a directory and returns the raw text alongside
its source path so later stages can cite where an answer came from.
"""

from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_documents(directory: str) -> list[tuple[str, str]]:
    """Load all supported files under `directory`.

    Args:
        directory: Folder to search recursively.

    Returns:
        A list of (text, source_path) tuples, one per non-empty file.
    """
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    documents: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_file(path)
        if text.strip():
            documents.append((text, str(path)))
    return documents


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    # errors="ignore" so a stray bad byte never crashes the whole index run.
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # extract_text() can return None for image-only pages.
    return "\n".join(page.extract_text() or "" for page in reader.pages)


if __name__ == "__main__":
    docs = load_documents("data")
    print(f"Loaded {len(docs)} document(s):")
    for text, source in docs:
        print(f"  - {source} ({len(text)} chars)")
