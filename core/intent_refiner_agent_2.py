import json
from typing import Dict, Any, Optional
from .state_types import GraphState
from .config import text_llm
from .prompt import INTENT_REFINEMENT_PROMPT
from .utils import extract_json_string, save_step_output

def node_refine_user_intent(state: GraphState) -> Dict[str, Any]:
    state["current_step"] = "refine_user_intent"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")
    user_query = state["user_query"] # 사용자의 원본 질문
    product_name = state.get("product_name_from_image", "알 수 없는 제품")
    image_data = state.get("image_data", {})
    image_claims = image_data.get("효능_주장", [])

    image_claims_example = ", ".join(image_claims[:2]) + (" 등" if len(image_claims) > 2 else "")

    prompt = INTENT_REFINEMENT_PROMPT.format(
        user_query=user_query,
        product_name=product_name,
        image_claims=str(image_claims),
        image_claims_example=image_claims_example
    )

    try:
        response = text_llm.invoke(prompt)
        raw_text = response.content # LLM의 원본 응답 텍스트

        # ❗ JSON 파싱 전에 Markdown 코드 블록 제거
        cleaned_json_text = extract_json_string(raw_text)
        parsed_response = json.loads(cleaned_json_text) # 정제된 텍스트로 JSON 파싱

        is_ambiguous = parsed_response.get("is_ambiguous", True) # 명시되지 않으면 모호한 것으로 간주
        inferred_query = parsed_response.get("inferred_query") # LLM이 추론한 단일 질문
        confidence = parsed_response.get("confidence_level", "정보 없음")
        reasoning = parsed_response.get("reasoning", "정보 없음")


        # downstream에서 사용할 최종 정제 쿼리 결정
        query_for_processing: str

        if not inferred_query:
            print(f"⚠️ LLM이 'inferred_query'를 생성하지 못했습니다. 원본 질문 '{user_query}'을(를) 내부 처리용으로 사용합니다.")
            query_for_processing = user_query
            is_ambiguous = True # 추론 실패 시 모호한 상태로 간주
        else:
            query_for_processing = inferred_query

        if is_ambiguous: # is_ambiguous는 LLM의 판단 또는 inferred_query 생성 실패 여부에 따라 결정
            if query_for_processing != user_query :
                print(f"🚦 사용자 원본 질문 '{user_query}'을(를) 내부적으로 다음과 같이 재구성하여 처리합니다.")
                print(f"🤖 추론된 내부 처리용 질문: '{query_for_processing}' (신뢰도: {confidence})")
                print(f"🤔 추론 근거: {reasoning}")
            else: # is_ambiguous가 True여도, 추론된 쿼리가 원본과 같을 수 있음 (LLM 판단) 또는 추론 실패로 원본 사용
                 print(f"🚦 사용자 원본 질문 '{user_query}'은(는) 모호하지만, 내부 처리 시 원본 질문을 사용합니다. (추론된 질문: '{query_for_processing}', 신뢰도: {confidence})")
                 if reasoning != "정보 없음": print(f"🤔 참고 근거: {reasoning}")

        else: # is_ambiguous가 False인 경우 (LLM이 원본 질문이 명확하다고 판단)
            print(f"👍 사용자 원본 질문 '{user_query}'을(를) 기반으로 내부 처리합니다.")
            # 이 경우에도 inferred_query는 원본을 약간 다듬은 형태일 수 있음
            if query_for_processing != user_query:
                print(f"⚙️ 내부 처리용 질문 (약간 수정됨): '{query_for_processing}' (신뢰도: {confidence})")
                if reasoning != "정보 없음": print(f"🤔 참고 근거: {reasoning}")
            else:
                print(f"⚙️ 내부 처리용 질문은 원본과 동일합니다: '{query_for_processing}'")


        output = {
            "is_query_ambiguous": is_ambiguous, # 원본 질문의 모호성 여부 (LLM 판단)
            "refined_user_query": query_for_processing, # 최종적으로 내부 처리에 사용될 질문
            "inferred_query_confidence": confidence,
            "inferred_query_reasoning": reasoning,
            "error_message": None
        }

        save_step_output(output, state["current_step"])
        return output

    except json.JSONDecodeError as e:
        error_msg = f"JSON 파싱 실패: {e}. 원본 응답: {response.content if 'response' in locals() else '응답 없음'}"
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"error_message": error_msg, "refined_user_query": user_query, "is_query_ambiguous": True}
    except Exception as e:
        error_msg = f"예상치 못한 오류 발생: {e}"
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"error_message": error_msg, "refined_user_query": user_query, "is_query_ambiguous": True}
