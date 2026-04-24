import torch
from sentence_transformers import SentenceTransformer

from pine import create_index_if_not_exists


print(torch.cuda.is_available(), torch.cuda.get_device_name())

model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_by_words(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks


with open("./el_quijote.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_by_words(text)

print(len(chunks))
print(chunks[0])


embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True, device="cuda")

print("Embeddings generados")


index = create_index_if_not_exists()

vectors = []
for i, emb in enumerate(embeddings):
    vectors.append(
        {"id": f"chunk-{i}", "values": emb.tolist(), "metadata": {"text": chunks[i]}}
    )

print("Subiendo a Pinecone...")
index.upsert(vectors=vectors)
print("Subida completada")


# test
query = "¿Quién es Don Quijote?"
query_embedding = model.encode([query], device="cuda")[0]

results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)

print("\nRESULTADOS:\n")
for match in results["matches"]:
    print(match["score"])
    print(match["metadata"]["text"][:200])
    print("-" * 50)
