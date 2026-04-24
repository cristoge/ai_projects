from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "el-quijote-rag"


def create_index_if_not_exists():
    existing_indexes = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Índice '{INDEX_NAME}' creado")
    else:
        print(f"Índice '{INDEX_NAME}' ya existe")

    return pc.Index(INDEX_NAME)


def get_index():
    return pc.Index(INDEX_NAME)
