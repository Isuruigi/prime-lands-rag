
from qdrant_client import QdrantClient

print("Creating client...")
client = QdrantClient("localhost", port=6333)

print("\nAttributes/Methods:")
for attr in dir(client):
    if not attr.startswith("_"):
        print(attr)

print("\nHas search?", hasattr(client, "search"))
print("\nHas query_points?", hasattr(client, "query_points"))
