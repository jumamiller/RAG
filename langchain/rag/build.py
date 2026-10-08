"""
LangChain RAG -- the same pipeline you built by hand in ../../rag/, but using
LangChain's built-in components.

Map to your from-scratch code:
    loader.py   -> DirectoryLoader + TextLoader   (stage 1)
    chunker.py  -> RecursiveCharacterTextSplitter (stage 2)
    embedder.py -> AzureOpenAIEmbeddings          (stage 3)
    store.py    -> FAISS                          (stage 4)
    pipeline.py -> a retrieval chain (LCEL)        (stage 5)

Run with:
    .\.venv\Scripts\python.exe .\langchain\rag\build.py

Fill in every TODO. Build stages 1-4 first, print the chunk count to confirm,
THEN add stage 5.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# The sample docs live in <repo>/data. This resolves it regardless of the
# directory you run the script from.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# --- Stage 1: LOAD ---------------------------------------------------------
# Replaces rag/loader.py.
# TODO: from langchain_community.document_loaders import DirectoryLoader, TextLoader
# TODO: loader = DirectoryLoader(str(DATA_DIR), glob="**/*.md", loader_cls=TextLoader)
# TODO: docs = loader.load()
# Note: each `doc` is a LangChain Document with .page_content and .metadata
#       (NOT your (text, source) tuples).


# --- Stage 2: SPLIT --------------------------------------------------------
# Replaces rag/chunker.py.
# TODO: from langchain_text_splitters import RecursiveCharacterTextSplitter
# TODO: splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# TODO: chunks = splitter.split_documents(docs)
# Think: this splits on CHARACTERS; your chunker split on WORDS. Why the default?


# --- Stage 3: EMBED --------------------------------------------------------
# Replaces rag/embedder.py.
# TODO: from langchain_openai import AzureOpenAIEmbeddings
# TODO: embeddings = AzureOpenAIEmbeddings(
#           azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
#       )
# (Auto-reads AZURE_OPENAI_ENDPOINT / _API_KEY / OPENAI_API_VERSION.)


# --- Stage 4: STORE --------------------------------------------------------
# Replaces rag/store.py. One call embeds every chunk AND builds the FAISS index.
# TODO: from langchain_community.vectorstores import FAISS
# TODO: store = FAISS.from_documents(chunks, embeddings)
# Think: your store.py L2-normalized vectors so inner-product == cosine.
#        Does this wrapper do that for you, or use a different metric (L2)?
#
# CHECKPOINT: print(len(chunks)) and print(chunks[0].page_content[:200]) here,
# run it, and confirm loading/splitting worked BEFORE writing stage 5.


# --- Stage 5: RETRIEVE + ANSWER -------------------------------------------
# Replaces rag/pipeline.py.
# TODO: retriever = store.as_retriever(search_kwargs={"k": 4})
#
# TODO: from langchain_openai import AzureChatOpenAI
# TODO: from langchain_core.prompts import ChatPromptTemplate
# TODO: from langchain.chains import create_retrieval_chain
# TODO: from langchain.chains.combine_documents import create_stuff_documents_chain
#
# TODO: build a ChatPromptTemplate with a {context} and {input} placeholder,
#       e.g. system: "Answer ONLY from the context. If unknown, say so.\n\n{context}"
#            human:  "{input}"
# TODO: llm = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"])
# TODO: doc_chain = create_stuff_documents_chain(llm, prompt)
# TODO: rag_chain = create_retrieval_chain(retriever, doc_chain)
#
# TODO: result = rag_chain.invoke({"input": "What are the two phases of RAG?"})
# TODO: print(result["answer"])
#       (result also has result["context"] -- the retrieved chunks. Inspect it!)


def main() -> None:
    raise NotImplementedError("Build stages 1-5 above, then wire them here.")


if __name__ == "__main__":
    main()
