"""Stages 5 + 6: retrieval and grounded generation.

`RAGPipeline` ties everything together:
  load -> chunk -> embed -> store     (indexing)
  embed query -> search -> prompt -> chat completion   (querying)

The system prompt forbids answering from outside the retrieved context -- that is
the line that separates RAG from a plain chatbot.
"""

import os

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .chunker import chunk_text
from .embedder import embed
from .loader import load_documents
from .store import VectorStore

EMBED_DIM = 1536  # text-embedding-3-small

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the answer is not contained in the context, say you "
    "don't know -- do not use outside knowledge. Cite the source file(s) you used."
)


class RAGPipeline:
    """Indexes a folder of documents and answers questions grounded in them."""

    def __init__(self, chat_deployment: str | None = None):
        self.store = VectorStore(EMBED_DIM)
        self.chat_deployment = (
            chat_deployment or os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        )
        self.client = AzureOpenAI()

    def index_directory(
        self, directory: str, chunk_size: int = 200, overlap: int = 40
    ) -> int:
        """Load, chunk, embed and store every document under `directory`.

        Returns the number of chunks indexed.
        """
        documents = load_documents(directory)

        chunks: list[str] = []
        sources: list[str] = []
        for text, source in documents:
            for chunk in chunk_text(text, chunk_size, overlap):
                chunks.append(chunk)
                sources.append(source)

        if not chunks:
            raise ValueError(f"No indexable content found in {directory!r}")

        vectors = embed(chunks)
        self.store.add(chunks, vectors, sources)
        return len(chunks)

    def ask(
        self, question: str, k: int = 4
    ) -> tuple[str, list[tuple[str, str, float]]]:
        """Retrieve the top-k chunks and generate a grounded answer.

        Returns (answer, hits) where each hit is (text, source, score).
        """
        query_vector = embed([question])[0]
        hits = self.store.search(query_vector, k=k)

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{self._format_context(hits)}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
        response = self.client.chat.completions.create(
            model=self.chat_deployment,
            messages=messages,
            temperature=0.0,  # deterministic, factual answers
        )
        answer = response.choices[0].message.content or ""
        return answer, hits

    @staticmethod
    def _format_context(hits: list[tuple[str, str, float]]) -> str:
        blocks = []
        for i, (text, source, score) in enumerate(hits, start=1):
            blocks.append(f"[{i}] (source: {source}, score: {score:.3f})\n{text}")
        return "\n\n".join(blocks) if blocks else "(no relevant context found)"
