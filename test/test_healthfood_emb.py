from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

vector_db = Chroma(
    persist_directory="./chroma_db",
    collection_name="health_collection",
    embedding_function=embedding,
)


retriever = vector_db.as_retriever(search_kwargs={"k": 20})
results = retriever.invoke("항산화")  # 범용 키워드

for doc in results:
    if doc.metadata.get("source") == "healthfood_claims_final10":
        print("✅ 찾음:")
        print("📄 page_content:", doc.page_content)
        print("📎 metadata:", doc.metadata)
        break
