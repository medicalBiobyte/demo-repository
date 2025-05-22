import os
import json
from datetime import datetime
from core.prompt import QUERY2KEYWORD_PROMPT
from core.config import vector_store, text_llm

SAVE_DIR = "RAG_RESULTS"
os.makedirs(SAVE_DIR, exist_ok=True)


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


def run_rag_from_ingredients(
    enriched_info: dict, user_query: str, strategy: str = "mmr", save: bool = True
) -> dict:
    """
    enriched_info["성분_효능"] 에 기반해 성분명별로 fnclty DB 검색 후 효능 일치 여부 평가
    """
    keywords = extract_keywords(user_query)
    ingredients = enriched_info.get("성분_효능", [])

    retriever = vector_store.as_retriever(
        search_type=strategy,
        search_kwargs=(
            {
                "k": 10,
                "fetch_k": 20,
                "lambda_mult": 0.1,
            }
            if strategy == "mmr"
            else {"k": 10}
        ),
    )

    evaluation_results = []
    for item in ingredients:
        name = item.get("성분명", "")
        if not name:
            continue

        docs = retriever.invoke(name)
        fnclty_docs = [doc for doc in docs if doc.metadata.get("source") == "fnclty"]

        if not fnclty_docs:
            evaluation_results.append(
                {
                    "성분명": name,
                    "효능": "정보 없음",
                    "일치도": "정보 없음",
                    "출처": "fnclty",
                }
            )
            continue

        best_doc = fnclty_docs[0]  # 가장 유사한 하나만 평가
        meta = best_doc.metadata
        efficacy_text = f"{meta.get('efficacy', '')} {meta.get('functionality', '')}".strip().lower()

        if not efficacy_text:
            match = "정보 없음"
        elif any(kw.lower() in efficacy_text for kw in keywords):
            match = "일치"
        else:
            match = "불일치"

        evaluation_results.append(
            {
                "성분명": name,
                "효능": efficacy_text if efficacy_text else "정보 없음",
                "일치도": match,
                "출처": "fnclty",
            }
        )

    final_decision = (
        "사용자 질문과 일부 성분의 효능이 일치합니다."
        if any(e["일치도"] == "일치" for e in evaluation_results)
        else "광고 주장의 근거가 부족합니다 (불일치)"
    )

    result = {
        "질문": user_query,
        "질문_키워드": keywords,
        "성분_기반_평가": evaluation_results,
        "최종_판단": final_decision,
    }

    # if save:
    #     safe_name = "_".join(keywords or ["query"]).replace(" ", "_")
    #     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_EVAL_{safe_name}.json"
    #     filepath = os.path.join(SAVE_DIR, filename)
    #     with open(filepath, "w", encoding="utf-8") as f:
    #         json.dump(result, f, ensure_ascii=False, indent=2)
    #     print(f"📁 평가 포함 RAG 결과 저장 완료 → {filepath}")

    return result


# 🧪 실행 예시
if __name__ == "__main__":
    from core.web_search_2 import get_enriched_product_info

    test_query = "이거 먹으면 키 크는데 효과 있나요?"
    enriched_info = get_enriched_product_info("키즈픽션")
    result = run_rag_from_ingredients(enriched_info, test_query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
