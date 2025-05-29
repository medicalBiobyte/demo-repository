from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

vector_db = Chroma(
    persist_directory="./chroma_db",
    collection_name="health_collection",
    embedding_function=embedding,
)

# Chroma에 저장된 전체 문서 중 일부 확인 (Chroma는 필터링을 지원하지 않음 → 검색어로 추정)
sample_queries = {
    # "fnclty": "콜레스테롤",  # page_content 예시 기반
    # "drug": "소화불량",
    "healthfood_claims_final10": "항산화",
}

for source, query in sample_queries.items():
    print(f"\n--- 🔍 Source: {source} | Query: {query} ---")
    retriever = vector_db.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke(query)
    for doc in results:
        if doc.metadata.get("source") == source:
            print("✅ page_content:\n", doc.page_content)
            print("📎 metadata:\n", doc.metadata)
        else:
            print("⚠️ source 불일치: ", doc.metadata.get("source"))
