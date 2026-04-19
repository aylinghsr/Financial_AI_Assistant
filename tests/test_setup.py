# save as test_setup.py and run with: python test_setup.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests

# Test Qdrant
client = QdrantClient(url="http://localhost:6333")
print("✅ Qdrant connected:", client.get_collections())

# Test embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode("test sentence")
print("✅ Embeddings working, shape:", emb.shape)

# Test Ollama
response = requests.post("http://localhost:11434/api/generate",
    json={"model": "mistral", "prompt": "say hello", "stream": False})
print("✅ Ollama working:", response.json()["response"][:50])