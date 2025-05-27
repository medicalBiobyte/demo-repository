import json
from typing import Dict, Any, Optional
from .state_types import GraphState
from .config import text_llm
from .prompt import INTENT_REFINEMENT_PROMPT
from .utils import extract_json_string, save_step_output

def node_refine_user_intent(state: GraphState) -> Dict[str, Any]:
    run_id = state.get("run_id") # run_id 가져오기
    current_step_name = "refine_user_intent"

    print(f"--- 🏃 [{run_id}] 단계 실행: {current_step_name} ---") # 로그에 run_id 추가

    user_query = state["user_query"]
    product_name = state.get("product_name_from_image", "알 수 없는 제품")
    image_data = state.get("image_data", {})
    image_claims = image_data.get("효능_주장", [])

    # 이 단계의 입력으로 저장될 정보 정의
    step_inputs_for_saving = {
        "user_query": user_query,
        "product_name_from_image": product_name,
        "image_claims": image_claims
    }

    node_return_output: Dict[str, Any] # 노드가 반환할 최종 딕셔너리
    status_for_saving = "success"      # 저장될 단계 실행 상태
    error_for_saving = None            # 저장될 오류 메시지

    try:
        image_claims_example = ", ".join(image_claims[:2]) + (" 등" if len(image_claims) > 2 else "")

        prompt = INTENT_REFINEMENT_PROMPT.format(
            user_query=user_query,
            product_name=product_name,
            image_claims=str(image_claims),
            image_claims_example=image_claims_example
        )

        response_from_llm = text_llm.invoke(prompt) # 변수명 변경 (response -> response_from_llm)
        raw_text = response_from_llm.content

        cleaned_json_text = extract_json_string(raw_text)
        parsed_response = json.loads(cleaned_json_text)

        is_ambiguous = parsed_response.get("is_ambiguous", True)
        inferred_query = parsed_response.get("inferred_query")
        confidence = parsed_response.get("confidence_level", "정보 없음")
        reasoning = parsed_response.get("reasoning", "정보 없음")

        query_for_processing: str
        if not inferred_query:
            print(f"⚠️ [{run_id}] LLM이 'inferred_query'를 생성하지 못했습니다. 원본 질문 '{user_query}'을(를) 내부 처리용으로 사용합니다.")
            query_for_processing = user_query
            is_ambiguous = True
        else:
            query_for_processing = inferred_query

        # ◀️ 로그에 run_id 추가
        if is_ambiguous:
            if query_for_processing != user_query :
                print(f"🚦 [{run_id}] 사용자 원본 질문 '{user_query}'을(를) 내부적으로 다음과 같이 재구성하여 처리합니다.")
                print(f"🤖 [{run_id}] 추론된 내부 처리용 질문: '{query_for_processing}' (신뢰도: {confidence})")
                print(f"🤔 [{run_id}] 추론 근거: {reasoning}")
            else:
                 print(f"🚦 [{run_id}] 사용자 원본 질문 '{user_query}'은(는) 모호하지만, 내부 처리 시 원본 질문을 사용합니다. (추론된 질문: '{query_for_processing}', 신뢰도: {confidence})")
                 if reasoning != "정보 없음": print(f"🤔 [{run_id}] 참고 근거: {reasoning}")
        else:
            print(f"👍 [{run_id}] 사용자 원본 질문 '{user_query}'을(를) 기반으로 내부 처리합니다.")
            if query_for_processing != user_query:
                print(f"⚙️ [{run_id}] 내부 처리용 질문 (약간 수정됨): '{query_for_processing}' (신뢰도: {confidence})")
                if reasoning != "정보 없음": print(f"🤔 [{run_id}] 참고 근거: {reasoning}")
            else:
                print(f"⚙️ [{run_id}] 내부 처리용 질문은 원본과 동일합니다: '{query_for_processing}'")

        # 성공 시 GraphState 업데이트를 위한 반환값 구성
        node_return_output = {
            "is_query_ambiguous": is_ambiguous,
            "refined_user_query": query_for_processing,
            "inferred_query_confidence": confidence, # GraphState에 이 필드들이 정의되어 있어야 함
            "inferred_query_reasoning": reasoning,   # GraphState에 이 필드들이 정의되어 있어야 함
            "error_message": None,
            "current_step": current_step_name
        }


    except json.JSONDecodeError as e:
        error_msg = f"JSON 파싱 실패: {e}. 원본 응답: {response_from_llm.content if 'response_from_llm' in locals() else '응답 없음'}"
        print(f"❌ [{run_id}] {current_step_name} 오류: {error_msg}")
        status_for_saving = "failure"
        error_for_saving = error_msg
        # 오류 발생 시 GraphState 업데이트를 위한 반환값 구성 (필수 필드에 기본값 제공)
        node_return_output = {
            "error_message": error_msg,
            "refined_user_query": user_query, # 원본 질문으로 대체
            "is_query_ambiguous": True,       # 모호한 것으로 간주
            "inferred_query_confidence": "오류",
            "inferred_query_reasoning": "JSON 파싱 오류",
            "current_step": current_step_name
        }
    except Exception as e:
        error_msg = f"예상치 못한 오류 발생 ({type(e).__name__}): {e}"
        print(f"❌ [{run_id}] {current_step_name} 오류: {error_msg}")
        status_for_saving = "failure"
        error_for_saving = error_msg
        # ◀️ 오류 발생 시 GraphState 업데이트를 위한 반환값 구성
        node_return_output = {
            "error_message": error_msg,
            "refined_user_query": user_query, # 원본 질문으로 대체
            "is_query_ambiguous": True,       # 모호한 것으로 간주
            "inferred_query_confidence": "오류",
            "inferred_query_reasoning": "예상치 못한 오류 발생",
            "current_step": current_step_name
        }

    # ◀️ 개선된 save_step_output 함수 호출 (항상 실행)
    save_step_output(
        run_id=run_id,
        step_name=current_step_name,
        step_inputs=step_inputs_for_saving,
        step_outputs=node_return_output, # 노드가 반환하는 전체 내용을 저장
        status=status_for_saving,
        error_message=error_for_saving
    )

    return node_return_output