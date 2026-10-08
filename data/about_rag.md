# What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a technique that combines a search step
with a language model. Instead of relying only on what the model memorized during
training, RAG retrieves relevant passages from an external knowledge base and
provides them to the model as context when generating an answer.

## Why use RAG?

- **Reduces hallucination.** The model answers from supplied context rather than guessing.
- **Uses private or fresh data.** You can answer questions about documents the model never saw during training.
- **Cited answers.** Because every answer is grounded in retrieved chunks, you can show the user the source.

## The two phases

RAG has two distinct phases. The **indexing** phase runs offline: documents are
loaded, split into chunks, embedded into vectors, and stored in a vector index.
The **query** phase runs per question: the question is embedded, the most similar
chunks are retrieved, and those chunks are inserted into the prompt for the model
to generate a grounded answer.

## Key parameters

The chunk size controls how much text goes into each passage. A common starting
point is a few hundred words per chunk with a small overlap so that ideas spanning
a boundary are not lost. The parameter *k* controls how many chunks are retrieved
for each question; a typical value is between 3 and 5.
