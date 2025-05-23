import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from datetime import datetime


# 기존 core 모듈 임포트
from core.text_extract_1 import extract_info_from_image
from core.web_search_2 import get_enriched_product_info
from core.claim_check_3 import get_product_evaluation
from core.rag_service_3_1 import run_rag_from_ingredients
from core.answer_user_4 import generate_natural_response

load_dotenv()


class GraphState(TypedDict):
    image_path: str
    user_query: str
    image_data: Optional[Dict[str, Any]]
    product_name_from_image: Optional[str]
    enriched_info: Optional[Dict[str, Any]]
    evaluation_result: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error_message: Optional[str]
    current_step: Optional[str]


# 공통 저장 함수 재정의 (코드 실행 상태 초기화로 인해)
def save_step_output(data: dict, step_name: str, folder: str = "STEP_OUTPUTS") -> None:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{step_name}.json"
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 저장 완료: {file_path}")


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

    return {
        "image_data": image_data_output,
        "product_name_from_image": product_name,
        "error_message": None,
    }


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
    user_query = state["user_query"]

    if not enriched_info:
        error_msg = "제품 평가를 위한 정보(enriched_info)가 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    evaluation_data = get_product_evaluation(enriched_info, user_query)

    if evaluation_data.get("최종_판단") == "광고 주장의 권가가 불쇄합니다 (불일치)":
        print("🔁 정확 매칭 실패 → RAG 보완 검색 실행")
        rag_result = run_rag_from_ingredients(enriched_info, user_query)
        evaluation_data["RAG_보완"] = rag_result

    if not evaluation_data or "최종_판단" not in evaluation_data:
        error_msg = "제품 평가에 실패했습니다."
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"evaluation_result": evaluation_data, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공.")

    output = {
        "evaluation_result": evaluation_data,
        "error_message": None,
    }
    save_step_output(output, "evaluate_product")  # 저장

    return {"evaluation_result": evaluation_data, "error_message": None}


def node_generate_natural_response(state: GraphState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    state["current_step"] = "generate_natural_response"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    evaluation_result = state["evaluation_result"]
    if not evaluation_result:
        error_msg = "자연어 답변 생성을 위한 평가 결과가 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    response_text = generate_natural_response(evaluation_result)

    if not response_text or not isinstance(response_text, str):
        error_msg = f"자연어 답변 생성에 실패했거나 결과가 문자열이 아닙니다. (타입: {type(response_text)})"
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
workflow.add_node("enrich_product_info", node_enrich_product_info)
workflow.add_node("evaluate_product", node_evaluate_product)
workflow.add_node("generate_response", node_generate_natural_response)
workflow.set_entry_point("extract_image_info")
workflow.add_edge("extract_image_info", "enrich_product_info")
workflow.add_edge("enrich_product_info", "evaluate_product")
workflow.add_edge("evaluate_product", "generate_response")
workflow.add_edge("generate_response", END)
app = workflow.compile()

if __name__ == "__main__":
    TEST_IMAGE_DIR = "img"
    test_image_filename = "height_medi_2.png"
    test_image_path = os.path.join(TEST_IMAGE_DIR, test_image_filename)
    sample_user_query = "이거 먹으면 키 크는데 효과 있나요?"

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
            print("\n---  최종 생성된 자연어 답변 ---")
            print(final_state["final_response"])
        else:
            print(
                "\n🤔 파이프라인은 완료되었지만, 최종 답변이 없거나 알 수 없는 문제가 발생했습니다."
            )
