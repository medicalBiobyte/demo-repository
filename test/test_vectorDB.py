import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import vector_store


def test_vector_store_with_query(query: str = "프로폴리스", k: int = 5):
    print(f"\n🔍 VectorDB에서 '{query}'로 검색 중 (Top-{k})...")
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    docs = retriever.invoke(query)

    if not docs:
        print("❌ 관련 문서를 찾을 수 없습니다.")
        return

    for i, doc in enumerate(docs):
        print(f"\n📄 문서 {i+1} --------------------------------")
        print("내용 (앞 200자):", doc.page_content[:200].replace("\n", " ") + "...")
        print("메타데이터:", doc.metadata)


if __name__ == "__main__":
    # 프로폴리스 테스트
    test_vector_store_with_query("프로폴리스")

    # 필요시 다른 키워드도 추가 테스트 가능
    # test_vector_store_with_query("밀크씨슬")
    # test_vector_store_with_query("비타민C")
