import os
import json
from datetime import datetime
from core.prompt import QUERY2KEYWORD_PROMPT
from core.config import vector_store, text_llm
from langchain.schema import Document  # 반드시 포함
import cohere

SAVE_DIR = "RAG_RESULTS"
os.makedirs(SAVE_DIR, exist_ok=True)
# 🧠 Cohere Reranker 설정
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(COHERE_API_KEY)


# 🔍 사용자 질문에서 키워드 추출
def extract_keywords(query: str) -> list[str]:
    prompt = QUERY2KEYWORD_PROMPT.replace("{query}", query)
    response = text_llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except Exception as e:
        print("❌ 키워드 JSON 파싱 실패:", e)
        print("원문:", response.content)
        return []


def decide_final_judgment(user_query: str, evaluation_results: list[dict]) -> str:
    formatted_result = "\n".join(
        [
            f"- 성분: {item['성분명']}\n  효능 요약: {item['효능']}\n  사용자 질문과 일치도: {item['일치도']}"
            for item in evaluation_results
        ]
    )
    prompt = f"""다음은 사용자의 건강 기능식품 관련 질문과 그에 대한 성분별 검색 결과입니다.

질문: "{user_query}"

성분별 평가 결과:
{formatted_result}

이 정보를 바탕으로, 이 제품의 광고 주장이 과학적으로 충분히 뒷받침되는지 종합적으로 판단해 주세요.
결론은 한 문장으로 간결하게 한국어로 작성해 주세요.

최종 판단:"""

    try:
        decision_response = text_llm.invoke(prompt)
        return decision_response.content.strip()
    except Exception as e:
        print(f"❌ LLM 판단 오류: {e}")
        return "판단 실패: 오류 발생"


def cohere_rerank(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    contents = [doc.page_content for doc in docs]
    response = cohere_client.rerank(
        query=query, documents=contents, top_n=top_n, model="rerank-multilingual-v3.0"
    )
    reranked = [docs[result.index] for result in response.results]
    return reranked


def run_rag_from_ingredients(
    enriched_info: dict, user_query: str, strategy: str = "mmr", save: bool = True
) -> dict:
    keywords = extract_keywords(user_query)
    ingredients = enriched_info.get("성분_효능", [])

    retriever = vector_store.as_retriever(
        search_type=strategy,
        search_kwargs=(
            {"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
            if strategy == "mmr"
            else {"k": 10}
        ),
    )

    evaluation_results = []
    seen_ingredients = set()  # ✅ 중복 방지용 집합

    for item in ingredients:
        ingredient_name = item.get("성분명", "")
        if not ingredient_name or ingredient_name in seen_ingredients:
            continue
        seen_ingredients.add(ingredient_name)

        print(f"🧬 RAG 검색 중 (성분: {ingredient_name})")
        retrieved_docs = retriever.invoke(ingredient_name)

        if not retrieved_docs:
            evaluation_results.append(
                {
                    "성분명": ingredient_name,
                    "효능": "정보 없음",
                    "일치도": "정보 없음",
                    "출처": ["문서 없음"],
                }
            )
            continue

        # 🔍 Rerank 전 출력
        print(f"🔎 [BEFORE RERANK] {ingredient_name} 관련 원본 문서:")
        for i, doc in enumerate(retrieved_docs[:5]):
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i+1}] {preview}...")

        # 💡 Cohere Rerank 적용
        reranked_docs = cohere_rerank(
            query=ingredient_name,
            docs=retrieved_docs,
            top_n=5,
        )

        # 🏆 Rerank 후 출력
        print(f"🏆 [AFTER RERANK] {ingredient_name} 관련 상위 문서:")
        for i, doc in enumerate(reranked_docs):
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i+1}] {preview}...")

        # 📦 출처 수집
        sources = []
        for doc in reranked_docs:
            source = doc.metadata.get("source", "출처 없음")
            identity = (
                doc.metadata.get("material")
                or doc.metadata.get("product_name")
                or doc.metadata.get("product")
                or "N/A"
            )
            sources.append(f"{source} / {identity}")

        context = "\n\n---\n\n".join(
            [
                f"[출처: {sources[i]}]\n{doc.page_content}"
                for i, doc in enumerate(reranked_docs)
            ]
        )

        prompt = f"""다음은 '{ingredient_name}' 성분과 관련된 문서들입니다.
이 문서들의 내용을 바탕으로, 주요 효능을 한글로 간결히 요약하세요.

[문서 시작]
{context}
[문서 끝]

효능 요약:"""

        try:
            llm_response = text_llm.invoke(prompt)
            efficacy = llm_response.content.strip() or "정보 없음"
            match_status = (
                "일치"
                if any(kw in efficacy for kw in keywords)
                else "불일치 또는 직접 관련 없음"
            )
        except Exception as e:
            print(f"❌ LLM 오류: {e}")
            efficacy = "정보 없음"
            match_status = "정보 없음"

        evaluation_results.append(
            {
                "성분명": ingredient_name,
                "효능": efficacy,
                "일치도": match_status,
                "출처": sources[:3],
                "원본문서": [doc.page_content for doc in retrieved_docs[:5]],
                "재정렬문서": [doc.page_content for doc in reranked_docs],
            }
        )

    final_decision = decide_final_judgment(user_query, evaluation_results)

    result = {
        "질문": user_query,
        "질문_키워드": keywords,
        "성분_기반_평가": evaluation_results,
        "최종_판단": final_decision,
    }

    if save:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_RAG_RERANK_RESULT.json"
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📁 결과 저장 완료: {filepath}")

    return result


# 🧪 실행 예시
if __name__ == "__main__":
    from core.web_search_3 import get_enriched_product_info

    print("🧪 RAG 서비스 직접 테스트 시작 🧪")
    print("=" * 50)

    print("\n[테스트 시나리오 1: '밀크씨슬(실리마린)' 성분으로 RAG 검색]")

    test_user_query_for_milk_thistle = (
        "밀크씨슬이 간 건강에 어떤 효과가 있나요? 광고처럼 정말 좋은가요?"
    )

    mock_enriched_info_milk_thistle = {
        "제품명": "가상 밀크씨슬 제품",
        "성분_효능": [
            {
                "성분명": "밀크씨슬(실리마린)",
                "효능": "간 건강 개선 (웹 정보 가정)",
                "출처": "가상 웹사이트",
            },
        ],
        "확정_성분": ["밀크씨슬(실리마린)"],
        "요약": "이것은 '밀크씨슬(실리마린)' 성분의 RAG 검색을 테스트하기 위한 가상 제품 정보입니다.",
    }

    print(f"사용자 질문: {test_user_query_for_milk_thistle}")
    print(
        f"입력 enriched_info의 성분: {[item['성분명'] for item in mock_enriched_info_milk_thistle.get('성분_효능', [])]}"
    )

    rag_result_milk_thistle = run_rag_from_ingredients(
        enriched_info=mock_enriched_info_milk_thistle,
        user_query=test_user_query_for_milk_thistle,
        save=False,
    )

    print("\n--- RAG 결과 (밀크씨슬 시나리오) ---")
    print(json.dumps(rag_result_milk_thistle, ensure_ascii=False, indent=2))
    print("-" * 50)
