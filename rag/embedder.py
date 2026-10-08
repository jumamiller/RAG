import os

from dotenv import load_dotenv
from openai import AzureOpenAI

# Load variables from a local .env file into the environment.
# Must run before we read os.environ below.
load_dotenv()

AZURE_OPENAI_EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using Azure OpenAI embeddings.

    Args:
        texts (list[str]): A list of texts to embed.

    Returns:
        list[list[float]]: A list of embeddings corresponding to the input texts.
    """
    # Initialize the AzureOpenAI client.
    # Reads AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY from the environment;
    # api_version comes from OPENAI_API_VERSION (or pass api_version=... explicitly).
    client = AzureOpenAI()

    # Create embeddings for the input texts (one batched API call).
    response = client.embeddings.create(
        model=AZURE_OPENAI_EMBED_DEPLOYMENT,
        input=texts,
    )

    # Extract and return the embeddings (attribute access on Pydantic objects).
    embeddings = [item.embedding for item in response.data]
    return embeddings


if __name__ == "__main__":
    import numpy as np

    def cosine(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

    sentences = [
        "The cat sat on the warm windowsill.",
        "A kitten napped in the sunny window.",
        "Quarterly tax filings are due next week.",
    ]
    vectors = embed(sentences)

    print(f"vector length: {len(vectors[0])}")  # expect 1536
    print(f"related   (0,1): {cosine(vectors[0], vectors[1]):.3f}")
    print(f"unrelated (0,2): {cosine(vectors[0], vectors[2]):.3f}")