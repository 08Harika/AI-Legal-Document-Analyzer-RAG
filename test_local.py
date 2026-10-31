
from langchain_ollama import OllamaEmbeddings

print("Testing local embedding model...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector = embeddings.embed_query(
    "This is a rental agreement."
)

print("Embedding model working successfully!")
print("Vector length:", len(vector))
print("First five values:", vector[:5])
