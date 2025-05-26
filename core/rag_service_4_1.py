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
    # enriched_info의 "성분_효능"은 [{ "성분명": "A", "효능": "웹검색효능1"}, ...] 형태일 것으로 예상
    ingredients_from_enriched_info = enriched_info.get("성분_효능", [])

    retriever = vector_store.as_retriever(
        search_type=strategy,
        search_kwargs=(
            {
                # MMR 경우, k는 최종 반환 문서 수, fetch_k는 초기 검색 문서 수
                "k": 5, # LLM에 전달할 최종 문서 수를 줄여서 컨텍스트 길이 관리 (예: 3~5개)
                "fetch_k": 20,
                "lambda_mult": 0.7, # 다양성 증진 (0.0 ~ 1.0, 높을수록 다양성)
            }
            if strategy == "mmr"
            else {"k": 5} # 유사도 검색 시에도 반환 문서 수 조절
        ),
    )

    evaluation_results = []
    for item in ingredients_from_enriched_info:
        ingredient_name_from_web = item.get("성분명", "") # 웹 검색에서 온 성분명
        # web_efficacy = item.get("효능", "") # 웹 검색에서 온 효능 (참고용)

        if not ingredient_name_from_web:
            continue

        # RAG 검색 시 성분명만 사용
        rag_query_text = ingredient_name_from_web
        print(f"🧬 RAG 검색 중 (사용자 질문 키워드: '{keywords}', 검색 성분명: '{ingredient_name_from_web}')")
        
        retrieved_docs = retriever.invoke(rag_query_text)
        
        extracted_efficacy_from_rag = "정보 없음"
        rag_source_info = "정보 없음" # RAG 문서 출처 초기화

        if retrieved_docs:
            print(f"📄 '{ingredient_name_from_web}'에 대해 {len(retrieved_docs)}개의 RAG 문서 찾음.")
            # 찾은 문서들의 page_content를 LLM에 전달할 컨텍스트로 구성
            # 너무 많은 문서를 한 번에 전달하면 LLM의 컨텍스트 길이 제한에 걸릴 수 있으므로, 상위 몇 개만 사용 (retriever의 k로 조절됨)
            context_for_llm = "\n\n---\n\n".join(
                [f"문서 출처: {doc.metadata.get('source', '알 수 없음')}\n문서 내용: {doc.page_content}" for doc in retrieved_docs]
            )
            
            # LLM에게 효능 정보 추출/요약 요청
            # 프롬프트는 필요에 따라 더 정교하게 수정 가능
            prompt_for_rag_efficacy_extraction = f"""다음은 '{ingredient_name_from_web}' 성분과 관련된 문서들입니다.
이 문서들의 내용을 바탕으로, '{ingredient_name_from_web}' 성분의 주요 효능 또는 기능성 내용을 한글로 간결하게 요약해 주십시오.
효능/기능성 내용이 여러가지일 경우, 가장 중요하거나 대표적인 것을 중심으로 언급하거나, 간략히 나열할 수 있습니다.
문서에서 관련 정보를 명확히 찾을 수 없다면 '정보 없음'이라고 답변해 주십시오.

[관련 문서 시작]
{context_for_llm}
[관련 문서 끝]

'{ingredient_name_from_web}'의 주요 효능/기능성 내용 요약:"""
            
            try:
                llm_response = text_llm.invoke(prompt_for_rag_efficacy_extraction)
                extracted_efficacy_from_rag = llm_response.content.strip()
                if not extracted_efficacy_from_rag or extracted_efficacy_from_rag.lower() == "정보 없음":
                    extracted_efficacy_from_rag = "정보 없음" # 일관된 표현 사용
                print(f"💡 LLM 추출 효능 ('{ingredient_name_from_web}'): {extracted_efficacy_from_rag}")

                # 출처 정보 (예: 가장 관련도 높은 문서의 출처 또는 여러 출처 요약)
                # 여기서는 간단히 첫 번째 문서의 출처를 사용하거나, "다수 출처" 등으로 표기 가능
                if retrieved_docs[0].metadata.get("source"):
                    rag_source_info = retrieved_docs[0].metadata.get("source")
                else:
                    rag_source_info = "출처 정보 없음"

            except Exception as e:
                print(f"❌ LLM으로 RAG 효능 추출 중 오류: {e}")
                extracted_efficacy_from_rag = "정보 없음 (추출 오류)"
                rag_source_info = "오류로 출처 확인 불가"
        else:
            print(f"ℹ️ '{ingredient_name_from_web}'에 대한 RAG 문서 없음.")
            # 문서가 아예 없으면 출처도 '정보 없음'으로 유지

        # 사용자 질문의 키워드와 RAG에서 추출된 효능 간의 일치도 판단
        match_status = "정보 없음" # 기본값
        if extracted_efficacy_from_rag not in ["정보 없음", "정보 없음 (추출 오류)"]:
            # 키워드 매칭: 추출된 효능 텍스트 내에 사용자 질문 키워드가 있는지 확인
            if keywords and any(kw.lower() in extracted_efficacy_from_rag.lower() for kw in keywords):
                match_status = "일치"
            else:
                # 키워드가 없거나, 효능은 있지만 키워드와 직접적 일치하지 않는 경우
                match_status = "불일치 또는 직접 관련 없음" 
        
        evaluation_results.append(
            {
                "성분명": ingredient_name_from_web,
                "효능": extracted_efficacy_from_rag, # LLM이 추출/요약한 효능
                "일치도": match_status,
                "출처": rag_source_info, # RAG 문서에서 가져온 출처
            }
        )

    # 최종 판단 로직 (기존과 유사하게 유지 또는 개선 가능)
    if any(e["일치도"] == "일치" for e in evaluation_results):
        final_decision = "사용자 질문과 RAG 정보 기반으로 일부 성분의 효능이 일치합니다."
    elif any(e["효능"] not in ["정보 없음", "정보 없음 (추출 오류)"] for e in evaluation_results): # 효능 정보는 있으나 질문과 불일치
        final_decision = "RAG 정보에 따르면, 일부 성분의 효능이 사용자 질문과 직접적으로 일치하지 않습니다."
    else: # RAG에서도 관련 효능 정보를 전혀 찾지 못한 경우
        final_decision = "RAG 정보에서도 광고 주장을 뒷받침할 근거를 찾기 어렵습니다."


    result = {
        "질문": user_query, # 원본 사용자 질문 또는 정제된 질문 (context에 따라 결정)
        "질문_키워드": keywords,
        "성분_기반_평가": evaluation_results,
        "최종_판단": final_decision,
    }

    if save:
        # 파일명에 사용할 안전한 문자열 생성 (키워드가 없을 경우 "rag_query" 사용)
        safe_name_parts = [kw.replace(" ", "_") for kw in keywords if kw] if keywords else ["rag_query"]
        safe_name = "_".join(safe_name_parts)
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_RAG_EVAL_{safe_name}.json"
        filepath = os.path.join(SAVE_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"📁 RAG 평가 결과 저장 완료 → {filepath}")
        except Exception as e:
            print(f"❌ RAG 평가 결과 저장 실패: {e}")
            
    return result


# 🧪 실행 예시
if __name__ == "__main__":
    from core.web_search_3 import get_enriched_product_info

    print("🧪 RAG 서비스 직접 테스트 시작 🧪")
    print("=" * 50)

    # --- 시나리오 1: "밀크씨슬(실리마린)" 성분에 대한 직접 테스트 ---
    print("\n[테스트 시나리오 1: '밀크씨슬(실리마린)' 성분으로 RAG 검색]")
    
    # "밀크씨슬" 관련 사용자 질문 예시
    test_user_query_for_milk_thistle = "밀크씨슬이 간 건강에 어떤 효과가 있나요? 광고처럼 정말 좋은가요?"
    
    # "밀크씨슬(실리마린)"을 포함하는 `enriched_info` 구조를 직접 구성합니다.
    # 이 구조는 `get_enriched_product_info` 함수의 일반적인 반환 형태와 유사하게 만듭니다.
    mock_enriched_info_milk_thistle = {
        "제품명": "가상 밀크씨슬 제품", # 테스트용 가상 제품명
        "성분_효능": [ # run_rag_from_ingredients 함수가 이 리스트를 사용합니다.
            {
                "성분명": "밀크씨슬(실리마린)", # RAG에서 검색을 시작할 성분명
                "효능": "간 건강 개선 (웹 정보 가정)", # 이 부분은 웹 검색 결과라고 가정 (참고용)
                "출처": "가상 웹사이트"
            },
            # 필요하다면 테스트를 위해 다른 가상 성분을 추가할 수 있습니다.
            # {
            #     "성분명": "코엔자임 Q10", 
            #     "효능": "항산화 작용 (웹 정보 가정)",
            #     "출처": "가상 웹사이트"
            # }
        ],
        "확정_성분": ["밀크씨슬(실리마린)"], # 기타 필요한 필드들
        "요약": "이것은 '밀크씨슬(실리마린)' 성분의 RAG 검색을 테스트하기 위한 가상 제품 정보입니다.",
    }

    print(f"사용자 질문: {test_user_query_for_milk_thistle}")
    print(f"입력 enriched_info의 성분: {[item['성분명'] for item in mock_enriched_info_milk_thistle.get('성분_효능', [])]}")

    # run_rag_from_ingredients 함수 호출 (수정된 버전 사용)
    rag_result_milk_thistle = run_rag_from_ingredients(
        enriched_info=mock_enriched_info_milk_thistle,
        user_query=test_user_query_for_milk_thistle,
        save=False # 테스트 중에는 파일 저장을 꺼도 됩니다.
    )
    
    print("\n--- RAG 결과 (밀크씨슬 시나리오) ---")
    # JSON 출력을 위해 json 모듈이 임포트 되어 있어야 합니다. (파일 상단에 import json)
    print(json.dumps(rag_result_milk_thistle, ensure_ascii=False, indent=2))
    print("-" * 50)