from core.config import vector_store # ChromaDB 인스턴스 (core/config.py 에서 가져옴)

print("--- 🧪 벡터 DB 직접 검색 테스트 ---")

# 테스트할 검색어들
queries_to_check = [
    "밀크씨슬(실리마린)", # 현재 RAG 서비스에서 사용하는 검색어
    "밀크씨슬",           # 좀 더 일반적인 용어
    "밀크씨슬 (카르두스 마리아누스) 추출물", # CSV에 있는 정확한 표현
    "간 건강에 도움을 줄 수 있음" # CSV에 있는 기능성 내용의 일부
]

for query in queries_to_check:
    print(f"\n--- 🔍 검색어: '{query}' ---")
    
    # retriever 설정을 rag_service_4_1.py와 유사하게 맞춰볼 수 있습니다.
    # 여기서는 기본적인 similarity_search를 사용하거나, MMR retriever를 사용할 수 있습니다.
    # docs = vector_store.similarity_search(query, k=3) 
    
    retriever = vector_store.as_retriever(
        search_type="similarity", # 또는 "mmr"
        search_kwargs={"k": 3}    # 상위 3개 문서 확인
    )
    docs = retriever.invoke(query)

    if docs:
        for i, doc in enumerate(docs):
            print(f"  📄 문서 {i+1}")
            print(f"     내용 (일부): {doc.page_content[:300]}...") # 저장된 page_content 확인
            print(f"     메타데이터: {doc.metadata}") # 저장된 metadata 확인 (특히 source)
            print("     ---")
    else:
        print("  ❌ 해당 검색어에 대한 문서를 찾을 수 없습니다.")