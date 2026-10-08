"""
Reference LangChain RAG -- a complete, runnable version.
(Your own work-in-progress lives in build.py; this is the finished comparison.)

Same five stages as your from-scratch ../../rag/ pipeline:
    load -> split -> embed -> store(FAISS) -> retrieve+answer

Run with:
    .\.venv\Scripts\python.exe .\langchain\rag\rag_demo.py
    .\.venv\Scripts\python.exe .\langchain\rag\rag_demo.py "your question here"
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def build_chain():
    # --- Stage 1: LOAD (replaces rag/loader.py) ---------------------------
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # --- Stage 2: SPLIT (replaces rag/chunker.py) -------------------------
    # Character-based splitting; the "Recursive" part tries paragraph -> line
    # -> word -> char boundaries so it breaks at natural seams when possible.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"[rag] loaded {len(docs)} doc(s) -> {len(chunks)} chunk(s)")

    # --- Stage 3: EMBED (replaces rag/embedder.py) ------------------------
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
    )

    # --- Stage 4: STORE (replaces rag/store.py) ---------------------------
    # One call embeds every chunk and builds the in-memory FAISS index.
    store = FAISS.from_documents(chunks, embeddings)

    # --- Stage 5: RETRIEVE + ANSWER (replaces rag/pipeline.py) ------------
    # LangChain 1.x removed the legacy create_retrieval_chain helpers, so we
    # compose the pipeline ourselves with LCEL: prompt | llm | parser.
    retriever = store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer strictly from the provided context. "
                "If the answer is not in the context, say you don't know.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
    )

    # The generation step: messages -> model -> plain string.
    answer_chain = prompt | llm | StrOutputParser()
    return retriever, answer_chain


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def ask(question: str, retriever, answer_chain):
    # 1. retrieve relevant chunks, 2. stuff them into the prompt, 3. generate.
    docs = retriever.invoke(question)
    answer = answer_chain.invoke({"context": format_docs(docs), "input": question})
    return answer, docs


def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the two phases of RAG and what is a typical value for k?"
    )

    retriever, answer_chain = build_chain()
    answer, docs = ask(question, retriever, answer_chain)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== SOURCES (retrieved chunks) ===")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"[{i}] {source}: {preview}...")


if __name__ == "__main__":
    main()
