from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from vector import embedding_model
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

INDEX_NAME = "el-quijote-rag"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Índice '{INDEX_NAME}' creado")
loader = TextLoader("./el_quijote.txt", encoding="utf-8")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)
chunks = splitter.split_documents(documents)
print(f"Total de chunks: {len(chunks)}")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=INDEX_NAME,
)
print("Subida completada")
results = vectorstore.similarity_search("¿Quién es Don Quijote?", k=5)
for doc in results:
    print(doc.page_content[:200])
    print("-" * 50)
