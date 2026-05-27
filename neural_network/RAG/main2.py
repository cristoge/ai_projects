from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector import get_vectorstore
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOllama(model="gemma4:e4b", base_url=OLLAMA_BASE_URL)

prompt = ChatPromptTemplate.from_template("""
Responde la pregunta usando solo el siguiente contexto:
{context}
Pregunta: {question}
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "¿Quién es Don Quijote?"
print(chain.invoke(question))
