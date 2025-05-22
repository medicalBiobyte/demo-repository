# rag_service_3_1.py

import json
from prompt import QUERY2KEYWORD_PROMPT
from config import vector_store, text_llm


def extract_keywords(query: str) -> list[str]:
    """사용자 질문에서 핵심 키워드 추출"""
    prompt = QUERY2KEYWORD_PROMPT.replace("{query}", query)
    response = text_llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except Exception as e:
        print("❌ 키워드 JSON 파싱 실패:", e)
        print("원문:", response.content)
        return []


def run_rag(query: str, strategy: str = "mmr") -> dict:
    """
    RAG 기반 검색 실행
    - strategy: mmr, similarity, score_threshold 등 선택 가능
    - 느슨한 MMR 설정 + efficacy/functionality 필터링
    """
    keywords = extract_keywords(query)
    combined_query = " ".join(keywords) or query

    retriever = vector_store.as_retriever(
        search_type=strategy,
        search_kwargs=(
            {
                "k": 20,
                "fetch_k": 40,
                "lambda_mult": 0.1,
            }
            if strategy == "mmr"
            else {"k": 20}
        ),
    )

    results = retriever.invoke(combined_query)

    # ✅ 핵심 기능성 필터링
    def is_relevant(doc, keywords: list[str]) -> bool:
        meta = doc.metadata
        text = f"{meta.get('efficacy', '')} {meta.get('functionality', '')}".lower()
        return any(kw.lower() in text for kw in keywords)

    filtered_results = [doc for doc in results if is_relevant(doc, keywords)]

    return {
        "질문": query,
        "추출_키워드": keywords,
        "검색된_문서": [
            {
                "내용": doc.page_content,
                "메타데이터": doc.metadata,
            }
            for doc in filtered_results[:3]  # 최대 3개
        ],
    }


# 🧪 실행 예시
if __name__ == "__main__":
    test_query = "이 약 먹으면 키에 도움이 되나요?"
    result = run_rag(test_query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
