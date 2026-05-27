from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
INDEX_NAME = "el-quijote-rag"

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text-v2-moe", base_url=OLLAMA_BASE_URL, dimensions=512
)


def get_vectorstore():
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding_model,
    )
