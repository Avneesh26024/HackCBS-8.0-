# utils.py

import os
from dotenv import load_dotenv

load_dotenv()
from typing import List
from openai import OpenAI

API_KEY = os.environ.get("Open_router_embedder_API_KEY")

if not API_KEY:
    print("Open_router_embedder_API_KEY not found in .env file.")
    # We don't exit, in case the key is set in the environment

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)


def get_embedding(texts: List[str]) -> List[List[float]]:
    """
    Gets embeddings for a list of texts using OpenRouter.
    """
    embeddings = []

    # Ensure all texts are non-empty strings
    processed_texts = [text if (text and text.strip()) else " " for text in texts]

    try:
        embedding_response = client.embeddings.create(
            model="google/gemini-embedding-001",
            input=processed_texts,
            encoding_format="float"
        )
        embeddings = [item.embedding for item in embedding_response.data]
        return embeddings
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a list of zero vectors with the correct dimensions
        # to avoid breaking ChromaDB (Gemini embedding dim is 768)
        return [[0.0] * 768 for _ in texts]

# The 'summarize_and_embed_schema' function has been removed
# as its logic is now handled directly by 'embedding_manager.py'
# which is more aligned with the 2-collection RAG approach.