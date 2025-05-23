import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from core.config import text_llm # text_llm 가져오기
from core.state_types import GraphState
from core.utils import extract_json_string, save_step_output

# 기존 core 모듈 임포트
from core.text_extract_1 import extract_info_from_image, extract_json_string
from core.intent_refiner_agent_2 import node_refine_user_intent
from core.web_search_3 import get_enriched_product_info
from core.claim_check_4 import get_product_evaluation
from core.rag_service_4_1 import run_rag_from_ingredients
from core.data_validator_agent_5 import node_validate_data_consistency
from core.answer_user_6 import generate_natural_response

load_dotenv()

def node_extract_image_info(state: GraphState) -> Dict[str, Any]:
    state["current_step"] = "extract_image_info"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")
    image_path = state["image_path"]

    if not os.path.exists(image_path):
        error_msg = f"이미지 파일을 찾을 수 없습니다: {image_path}"
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    image_data_output = extract_info_from_image(image_path)

    if (
        not image_data_output
        or "제품명" not in image_data_output
        or not image_data_output.get("제품명")
    ):
        error_msg = "이미지에서 유효한 정보를 추출하지 못했습니다."
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        if image_data_output:
            print(json.dumps(image_data_output, indent=2, ensure_ascii=False))
        return {"image_data": image_data_output, "error_message": error_msg}

    product_name = image_data_output.get("제품명", "").split("/")[0].strip()
    if not product_name:
        error_msg = "추출된 제품명이 유효하지 않습니다."
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"image_data": image_data_output, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공. 제품명: {product_name}")

    output = {
        "image_data": image_data_output,
        "product_name_from_image": product_name,
        "error_message": None,
    }
    save_step_output(output, "extract_image_info")  # 저장

    return output

def node_enrich_product_info(state: GraphState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    state["current_step"] = "enrich_product_info"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    product_name = state["product_name_from_image"]
    if not product_name:
        error_msg = "정보 보감을 위한 제품명이 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    enriched_data = get_enriched_product_info(product_name)

    if not enriched_data or enriched_data.get("error"):
        error_msg = f"웹 정보를 보감하지 못했습니다. 메시지: {enriched_data.get('error', '알 수 없음') if enriched_data else '데이터 없음'}"
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"enriched_info": enriched_data, "error_message": error_msg}

    if state.get("image_data"):
        enriched_data["original_효력_주장"] = state["image_data"].get("효력_주장")

    print(f"✅ {state['current_step']} 성공.")

    output = {
        "enriched_info": enriched_data,
        "error_message": None,
    }
    save_step_output(output, "enrich_product_info")  # 저장

    return {"enriched_info": enriched_data, "error_message": None}


def node_evaluate_product(state: GraphState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    state["current_step"] = "evaluate_product"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    enriched_info = state["enriched_info"]
    original_user_query = state["user_query"] # 사용자의 원본 질문
    # ❗ 내부 평가에는 정제된 질문을 사용
    query_for_evaluation = state.get("refined_user_query", original_user_query)

    if not enriched_info:
        error_msg = "제품 평가를 위한 정보(enriched_info)가 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    print(f" 평가에 사용될 질문: '{query_for_evaluation}' (원본 질문: '{original_user_query}')")
    # ❗ get_product_evaluation 호출 시 정제된 질문(query_for_evaluation)을 전달
    # ❗ 그리고 get_product_evaluation 함수가 반환하는 결과에 '사용자_질문' 필드로 original_user_query를 넣어주도록 수정 필요
    evaluation_data = get_product_evaluation(
        enriched_data=enriched_info,
        user_query=query_for_evaluation, # 내부 처리용 질문
        original_user_query_for_display=original_user_query # 최종 표기용 원본 질문 전달
    )

    # RAG 보완 로직 (기존과 동일하게 query_for_evaluation 사용)
    if evaluation_data.get("최종_판단", "").startswith("광고 주장의 근거가 부족합니다"): # 최종_판단 문자열 시작 부분으로 체크
        print(f"🔁 '{evaluation_data.get('제품명', '알수없음')}' 제품에 대한 초기 평가 결과 근거 부족, RAG 보완 검색 실행...")
        # run_rag_from_ingredients 호출 시에도 정제된 질문(query_for_evaluation) 사용
        rag_result = run_rag_from_ingredients(enriched_info, query_for_evaluation)
        if rag_result and rag_result.get("성분_기반_평가"):
            evaluation_data["RAG_보완"] = rag_result
            print(f"🔄 RAG 보완 결과: {rag_result.get('최종_판단')}")
        else:
            print(" RAG 보완 정보 없음 또는 유효하지 않음.")


    if not evaluation_data or "최종_판단" not in evaluation_data:
        error_msg = "제품 평가에 실패했습니다."
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"evaluation_result": evaluation_data, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공.")

    output = {
        "evaluation_result": evaluation_data,
        "error_message": None,
    }
    save_step_output(output, "evaluate_product")
    return output
    
def node_generate_natural_response(state: GraphState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    state["current_step"] = "generate_natural_response"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    evaluation_result = state["evaluation_result"]
    if not evaluation_result:
        error_msg = "답변 생성을 위한 평가 결과가 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    response_text = generate_natural_response(evaluation_result)

    if not response_text or not isinstance(response_text, str):
        error_msg = f"답변 생성에 실패했거나 결과가 문자열이 아닙니다. (타입: {type(response_text)})"
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"final_response": response_text, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공.")

    output = {
        "final_response": response_text,
        "error_message": None,
    }
    save_step_output(output, "generate_natural_response")  # 저장

    return {"final_response": response_text, "error_message": None}


workflow = StateGraph(GraphState)
workflow.add_node("extract_image_info", node_extract_image_info)
# 🆕 사용자 의도 분석 에이전트 노드 추가
workflow.add_node("refine_user_intent", node_refine_user_intent)
workflow.add_node("enrich_product_info", node_enrich_product_info)
workflow.add_node("evaluate_product", node_evaluate_product)
# 🆕 데이터 검증 에이전트 노드 추가
workflow.add_node("validate_data_consistency", node_validate_data_consistency)
workflow.add_node("generate_response", node_generate_natural_response)

workflow.set_entry_point("extract_image_info")

workflow.add_edge("extract_image_info", "refine_user_intent")
workflow.add_edge("refine_user_intent", "enrich_product_info")
workflow.add_edge("enrich_product_info", "evaluate_product")
workflow.add_edge("evaluate_product", "validate_data_consistency") # 제품 평가 후 데이터 검증
workflow.add_edge("validate_data_consistency", "generate_response") # 검증 후 최종 답변 생성
workflow.add_edge("generate_response", END)
app = workflow.compile()


if __name__ == "__main__":
    TEST_IMAGE_DIR = "img"
    test_image_filename = "milk_thistle_1.jpeg"
    test_image_path = os.path.join(TEST_IMAGE_DIR, test_image_filename)
    sample_user_query = "이거 먹으면 혈압에 좋나요?"

    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"🚨 오류: 이미지 디렉터리 '{TEST_IMAGE_DIR}'를 찾을 수 없습니다.")
        print("디렉토리를 생성하고 테스트 이미지를 넣어주세요.")
    elif not os.path.exists(test_image_path):
        print(f"🚨 오류: 테스트 이미지 '{test_image_path}'를 찾을 수 없습니다.")
        print(f"'{TEST_IMAGE_DIR}' 폴더에 '{test_image_filename}' 파일을 넣어주세요.")
    else:
        print("🚀 LangGraph 기반 통합 파이프라인 시작 🚀")

        initial_state = {"image_path": test_image_path, "user_query": sample_user_query}
        final_state = app.invoke(initial_state)

        print("\n--- 📈 최종 그래프 상태 ---")
        print(json.dumps(final_state, indent=2, ensure_ascii=False))
        print("-" * 50)

        if final_state.get("error_message"):
            print(f"\n🚫 파이프라인 중 오류 발생: {final_state['error_message']}")
            print(f"오류 발생 단계: {final_state.get('current_step', '알 수 없음')}")
        elif final_state.get("final_response"):
            print("\n🎉 파이프라인 성공적으로 완료! 🎉")
            print("\n---  최종 생성된 답변 ---")
            print(final_state["final_response"])
        else:
            print(
                "\n🤔 파이프라인은 완료되었지만, 최종 답변이 없거나 알 수 없는 문제가 발생했습니다."
            )
