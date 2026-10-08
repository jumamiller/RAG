"""CLI for the minimal RAG pipeline.

Usage:
    python main.py index               # index ./data and report chunk count
    python main.py ask "your question" # index ./data, then answer the question
"""

import argparse

from rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal Retrieval-Augmented Generation over local documents."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="Index documents and report counts.")
    index_parser.add_argument("--data", default="data", help="Documents directory.")

    ask_parser = sub.add_parser("ask", help="Index documents, then answer a question.")
    ask_parser.add_argument("question", help="The question to ask.")
    ask_parser.add_argument("--data", default="data", help="Documents directory.")
    ask_parser.add_argument(
        "-k", type=int, default=4, help="Number of chunks to retrieve."
    )

    args = parser.parse_args()
    pipeline = RAGPipeline()

    chunk_count = pipeline.index_directory(args.data)

    if args.command == "index":
        print(f"Indexed {chunk_count} chunk(s) from {args.data!r}.")
        return

    print(f"Indexed {chunk_count} chunk(s). Thinking...\n")
    answer, hits = pipeline.ask(args.question, k=args.k)

    print("Answer:\n" + answer + "\n")
    print("Sources:")
    seen: set[str] = set()
    for _text, source, score in hits:
        if source not in seen:
            print(f"  - {source} (score {score:.3f})")
            seen.add(source)


if __name__ == "__main__":
    main()