from pine import get_index
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = get_index()

while True:
    question = input("Pregunta: ")

    query_embedding = model.encode([question], device="cuda")[0]

    results = index.query(
        vector=query_embedding.tolist(), top_k=5, include_metadata=True
    )

    context = "\n\n".join([m["metadata"]["text"] for m in results["matches"]])

    print("Contexto")
    print(context)
    print("----------------------------------------------")
