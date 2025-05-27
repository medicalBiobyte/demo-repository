import json
from typing import Dict, Any, Optional, List

from .state_types import GraphState
from .config import text_llm
from .prompt import DATA_VALIDATION_PROMPT
from .utils import extract_json_string, save_step_output


def node_validate_data_consistency(state: GraphState) -> Dict[str, Any]:
    if state.get("error_message") and state.get("current_step") != "evaluate_product": # 이전 단계 에러 시 스킵 (평가 후에는 실행)
        return {}
    state["current_step"] = "validate_data_consistency"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    image_data = state.get("image_data", {})
    evaluation_result = state.get("evaluation_result", {}) # claim_check_3.py의 결과

    product_name = image_data.get("제품명", evaluation_result.get("제품명", "알 수 없음"))
    image_claims_list = image_data.get("효능_주장", [])

    # evaluation_result에서 정보 추출
    # claim_check_3.py 및 rag_service_3_1.py의 출력 구조를 참조합니다.
    ingredients_from_web_db = evaluation_result.get("확정_성분", [])
    matched_ingredients_eval_raw = evaluation_result.get("매칭_성분", []) # claim_check_3
    rag_based_eval_raw = evaluation_result.get("RAG_보완", {}).get("성분_기반_평가", []) # rag_service_3_1

    # 프롬프트에 넣기 좋게 문자열로 변환
    def format_eval(eval_list):
        return "\n".join([f"- {item.get('성분명')}: {item.get('효능')} (일치도: {item.get('일치도')}, 출처: {item.get('출처', 'DB/웹')})" for item in eval_list if item.get('성분명')])

    matched_ingredients_eval_str = format_eval(matched_ingredients_eval_raw)
    rag_based_eval_str = format_eval(rag_based_eval_raw)
    user_query_keywords = evaluation_result.get("질문_핵심_키워드", []) # claim_check_3

    prompt = DATA_VALIDATION_PROMPT.format(
        product_name_from_image=product_name,
        image_claims_list=str(image_claims_list),
        ingredients_from_web_db=str(ingredients_from_web_db),
        matched_ingredients_eval=matched_ingredients_eval_str if matched_ingredients_eval_str else "정보 없음",
        rag_based_eval=rag_based_eval_str if rag_based_eval_str else "정보 없음",
        user_query_keywords=str(user_query_keywords)
    )

    try:
        response = text_llm.invoke(prompt)
        raw_text = response.content # LLM의 원본 응답 텍스트

        # ❗ JSON 파싱 전에 Markdown 코드 블록 제거
        cleaned_json_text = extract_json_string(raw_text)
        # 정제된 텍스트로 JSON 파싱
        parsed_response = json.loads(cleaned_json_text)

        consistency_status = parsed_response.get("consistency_status", "정보 부족으로 판단 어려움")
        validation_details = parsed_response.get("validation_details", [])
        overall_assessment = parsed_response.get("overall_assessment", "") # overall_assessment도 가져오도록 추가

        print(f"📊 데이터 일관성 검증 결과: {consistency_status}")
        if validation_details:
            for detail in validation_details:
                if not detail.get("evidence_found"):
                    print(f"  ⚠️ 주장 불일치/근거 부족: '{detail.get('claim')}' - {detail.get('discrepancy_note')}")

        output = {
            "data_consistency_status": consistency_status,
            "validation_issues": validation_details,
            "error_message": None
        }
        
        if state.get("evaluation_result"):
            current_eval_result = state["evaluation_result"].copy() # 복사본 사용 권장
            current_eval_result["data_validation_summary"] = {
                "status": consistency_status,
                "issues": validation_details,
                "assessment_comment": overall_assessment # 파싱한 overall_assessment 사용
            }
            # output 딕셔너리를 직접 수정하거나, 반환 시 합치는 대신,
            # GraphState의 키에 직접 할당하는 방식으로 상태를 업데이트합니다.
            # 여기서는 evaluation_result를 업데이트하고, output에는 validation_issues 등만 포함
            return {
                "evaluation_result": current_eval_result, 
                "data_consistency_status": consistency_status, # GraphState 필드에 직접 매핑되도록
                "validation_issues": validation_details,       # GraphState 필드에 직접 매핑되도록
                "error_message": None
            }

        # evaluation_result가 없는 경우 (이론적으로는 이전 단계에서 채워져야 함)
        return output

    except json.JSONDecodeError as e:
        # ❗ 오류 발생 시 원본 응답(raw_text)과 정제 시도한 텍스트(cleaned_json_text)를 함께 로깅하면 디버깅에 도움
        cleaned_text_for_log = cleaned_json_text if 'cleaned_json_text' in locals() else "정제 전 또는 정제 실패"
        error_msg = f"JSON 파싱 실패: {e}. 원본 응답: {raw_text if 'raw_text' in locals() else '응답 없음'}. 정제 시도한 텍스트: {cleaned_text_for_log}"
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"error_message": error_msg} # GraphState의 error_message 필드에 직접 할당됨
    except Exception as e:
        error_msg = f"예상치 못한 오류 발생 ({type(e).__name__}): {e}"
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"error_message": error_msg}