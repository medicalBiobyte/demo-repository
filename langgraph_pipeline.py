import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from core.config import text_llm # text_llm 가져오기
from core.state_types import GraphState
from core.utils import extract_json_string, save_step_output, save_run_metadata
import uuid # run_id 생성을 위해 추가
from datetime import datetime # 타임스탬프를 위해 추가

# 기존 core 모듈 임포트
from core.text_extract_1 import extract_info_from_image
from core.intent_refiner_agent_2 import node_refine_user_intent
from core.web_search_3 import get_enriched_product_info
from core.claim_check_4 import get_product_evaluation
from core.rag_service_4_1 import run_rag_from_ingredients
from core.answer_user_5 import generate_natural_response

load_dotenv()

def node_extract_image_info(state: GraphState) -> Dict[str, Any]:
    run_id = state.get("run_id") # ◀️ run_id 가져오기
    current_step_name = "extract_image_info"
    # state["current_step"] = current_step_name # current_step은 마지막 상태에 자동으로 기록될 수 있도록 GraphState에 유지
    print(f"--- 🏃 [{run_id}] 단계 실행: {current_step_name} ---") # 로그에 run_id 추가
    
    image_path = state["image_path"]
    step_inputs_for_saving = {"image_path": image_path} # 이 단계의 입력 정의

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

    # 반환할 결과 (상태 업데이트용)
    node_return_output: Dict[str, Any]
    status_for_saving = "success"
    error_for_saving = None

    if not os.path.exists(image_path):
        error_msg = f"이미지 파일을 찾을 수 없습니다: {image_path}"
        print(f"❌ [{run_id}] 오류: {error_msg}")
        node_return_output = {"error_message": error_msg, "current_step": current_step_name}
        status_for_saving = "failure"
        error_for_saving = error_msg
    else:
        image_data_output = extract_info_from_image(image_path)
        if (
            not image_data_output
            or "제품명" not in image_data_output
            or not image_data_output.get("제품명")
        ):
            error_msg = "이미지에서 유효한 정보를 추출하지 못했습니다."
            print(f"❌ [{run_id}] {current_step_name} 오류: {error_msg}")
            if image_data_output:
                print(json.dumps(image_data_output, indent=2, ensure_ascii=False))
            node_return_output = {"image_data": image_data_output, "error_message": error_msg, "current_step": current_step_name}
            status_for_saving = "failure"
            error_for_saving = error_msg
        else:
            product_name = image_data_output.get("제품명", "").split("/")[0].strip()
            if not product_name:
                error_msg = "추출된 제품명이 유효하지 않습니다."
                print(f"❌ [{run_id}] {current_step_name} 오류: {error_msg}")
                node_return_output = {"image_data": image_data_output, "error_message": error_msg, "current_step": current_step_name}
                status_for_saving = "failure"
                error_for_saving = error_msg
            else:
                print(f"✅ [{run_id}] {current_step_name} 성공. 제품명: {product_name}")
                node_return_output = {
                    "image_data": image_data_output,
                    "product_name_from_image": product_name,
                    "error_message": None, # 성공 시 에러 메시지는 None
                    "current_step": current_step_name
                }

    # 개선된 save_step_output 호출
    save_step_output(
        run_id=run_id,
        step_name=current_step_name,
        step_inputs=step_inputs_for_saving,
        step_outputs=node_return_output, # 노드가 반환하는 전체 내용을 저장
        status=status_for_saving,
        error_message=error_for_saving
    )

    # LangGraph 상태 업데이트를 위해 반환 (error_message와 current_step 포함)
    return node_return_output

def node_enrich_product_info(state: GraphState) -> Dict[str, Any]:
    run_id = state.get("run_id")
    current_step_name = "enrich_product_info"
    
    step_inputs_for_saving = {
        "product_name_from_image": state.get("product_name_from_image"),
        "image_data_claims": state.get("image_data", {}).get("효능_주장")
    }
    
    node_return_output: Dict[str, Any]
    status_for_saving = "success"
    error_for_saving = None

    if state.get("error_message"):
        error_msg = f"이전 단계 오류로 인해 {current_step_name} 실행을 건너뜁니다: {state.get('error_message')}"
        print(f"🟡 [{run_id}] {error_msg}")
        status_for_saving = "skipped"
        error_for_saving = state.get("error_message") # 이전 오류를 기록
        node_return_output = {"error_message": error_for_saving, "current_step": state.get("current_step")} # 이전 단계의 current_step 유지
    else:
        print(f"--- 🏃 [{run_id}] 단계 실행: {current_step_name} ---")
        product_name = state.get("product_name_from_image")
        if not product_name:
            error_msg = "정보 보강을 위한 제품명이 없습니다."
            print(f"❌ [{run_id}] 오류: {error_msg}")
            status_for_saving = "failure"
            error_for_saving = error_msg
            node_return_output = {"error_message": error_msg, "current_step": current_step_name}
        else:
            enriched_data = get_enriched_product_info(product_name) # web_search_3.py의 함수
            if not enriched_data or enriched_data.get("error"):
                error_msg = f"웹 정보를 보강하지 못했습니다. 메시지: {enriched_data.get('error', '알 수 없음') if enriched_data else '데이터 없음'}"
                print(f"❌ [{run_id}] {current_step_name} 오류: {error_msg}")
                status_for_saving = "failure"
                error_for_saving = error_msg
                node_return_output = {"enriched_info": enriched_data, "error_message": error_msg, "current_step": current_step_name}
            else:
                if state.get("image_data"): # 원본 이미지의 효능 주장을 enriched_info에 추가
                    enriched_data["original_효력_주장"] = state.get("image_data", {}).get("효능_주장")
                print(f"✅ [{run_id}] {current_step_name} 성공.")
                node_return_output = {
                    "enriched_info": enriched_data,
                    "error_message": None,
                    "current_step": current_step_name
                }
    save_step_output(
        run_id=run_id,
        step_name=current_step_name,
        step_inputs=step_inputs_for_saving,
        step_outputs=node_return_output,
        status=status_for_saving,
        error_message=error_for_saving
    )
    return node_return_output


def node_evaluate_product(state: GraphState) -> Dict[str, Any]:
    run_id = state.get("run_id")
    current_step_name = "evaluate_product"

    step_inputs_for_saving = {
        "enriched_info": state.get("enriched_info"),
        "user_query": state.get("user_query"),
        "refined_user_query": state.get("refined_user_query")
    }

    node_return_output: Dict[str, Any]
    status_for_saving = "success"
    error_for_saving = None

    if state.get("error_message"): # 이전 단계 오류 확인
        error_msg = f"이전 단계 오류로 인해 {current_step_name} 실행을 건너뜁니다: {state.get('error_message')}"
        print(f"🟡 [{run_id}] {error_msg}")
        status_for_saving = "skipped"
        error_for_saving = state.get("error_message")
        node_return_output = {"error_message": error_for_saving, "current_step": state.get("current_step")}
    else:
        print(f"--- 🏃 [{run_id}] 단계 실행: {current_step_name} ---")
        enriched_info = state.get("enriched_info")
        original_user_query = state.get("user_query", "")
        query_for_evaluation = state.get("refined_user_query", original_user_query)

        if not enriched_info:
            error_msg = "제품 평가를 위한 정보(enriched_info)가 없습니다."
            print(f"❌ [{run_id}] 오류: {error_msg}")
            status_for_saving = "failure"
            error_for_saving = error_msg
            node_return_output = {"error_message": error_msg, "current_step": current_step_name}
        else:
            print(f" 평가에 사용될 질문: '{query_for_evaluation}' (원본 질문: '{original_user_query}')")
            evaluation_data = get_product_evaluation( # claim_check_4.py의 함수
                enriched_data=enriched_info,
                user_query=query_for_evaluation,
                original_user_query_for_display=original_user_query
            )

            if evaluation_data.get("최종_판단", "").startswith("광고 주장의 근거가 부족합니다"):
                product_name_for_log = evaluation_data.get('제품명', enriched_info.get('제품명', '알수없음'))
                print(f"🔁 [{run_id}] '{product_name_for_log}' 제품에 대한 초기 평가 결과 근거 부족, RAG 보완 검색 실행...")
                rag_result = run_rag_from_ingredients(enriched_info, query_for_evaluation) # rag_service_4_1.py의 함수
                if rag_result and rag_result.get("성분_기반_평가"):
                    evaluation_data["RAG_보완"] = rag_result
                    print(f"🔄 [{run_id}] RAG 보완 결과: {rag_result.get('최종_판단')}")
                else:
                    print(f" [{run_id}] RAG 보완 정보 없음 또는 유효하지 않음.")
            
            if not evaluation_data or "최종_판단" not in evaluation_data:
                error_msg = "제품 평가에 실패했습니다."
                print(f"❌ [{run_id}] {current_step_name} 오류: {error_msg}")
                status_for_saving = "failure"
                error_for_saving = error_msg
                node_return_output = {"evaluation_result": evaluation_data, "error_message": error_msg, "current_step": current_step_name}
            else:
                print(f"✅ [{run_id}] {current_step_name} 성공.")
                node_return_output = {
                    "evaluation_result": evaluation_data,
                    "error_message": None,
                    "current_step": current_step_name
                }

    save_step_output(
        run_id=run_id,
        step_name=current_step_name,
        step_inputs=step_inputs_for_saving,
        step_outputs=node_return_output,
        status=status_for_saving,
        error_message=error_for_saving
    )
    return node_return_output
    
def node_generate_natural_response(state: GraphState) -> Dict[str, Any]:
    run_id = state.get("run_id")
    current_step_name = "generate_natural_response" # 함수 이름과 일치

    step_inputs_for_saving = {"evaluation_result": state.get("evaluation_result")}

    node_return_output: Dict[str, Any]
    status_for_saving = "success"
    error_for_saving = None
    
    # 이전 단계 오류 확인 (validate_data_consistency가 빠졌으므로 evaluate_product의 오류를 직접 체크)
    if state.get("error_message") and state.get("current_step") != "evaluate_product": # 만약 evaluate_product 자체가 실패했다면
         # generate_response는 오류가 있어도 요약은 시도해볼 수 있으므로, current_step 체크를 더 면밀히 할 수 있음
         # 여기서는 간단히 이전 오류가 있으면 스킵하는 로직으로 처리 (필요시 수정)
        error_msg = f"이전 단계 오류로 인해 {current_step_name} 실행을 건너뜁니다: {state.get('error_message')}"
        print(f"🟡 [{run_id}] {error_msg}")
        status_for_saving = "skipped"
        error_for_saving = state.get("error_message")
        node_return_output = {"error_message": error_for_saving, "current_step": state.get("current_step")}
    elif not state.get("evaluation_result"): # evaluation_result 자체가 없는 경우
        error_msg = "답변 생성을 위한 평가 결과가 없습니다."
        print(f"❌ [{run_id}] {current_step_name} 오류: {error_msg}")
        status_for_saving = "failure"
        error_for_saving = error_msg
        node_return_output = {"error_message": error_msg, "current_step": current_step_name} # 이 단계에서 오류 발생
    else:
        print(f"--- 🏃 [{run_id}] 단계 실행: {current_step_name} ---")
        evaluation_result = state["evaluation_result"]
        response_text = generate_natural_response(evaluation_result) # answer_user_6.py의 함수

        if not response_text or not isinstance(response_text, str):
            error_msg = f"답변 생성에 실패했거나 결과가 문자열이 아닙니다. (타입: {type(response_text)})"
            print(f"❌ [{run_id}] {current_step_name} 오류: {error_msg}")
            status_for_saving = "failure"
            error_for_saving = error_msg
            node_return_output = {"final_response": response_text, "error_message": error_msg, "current_step": current_step_name}
        else:
            print(f"✅ [{run_id}] {current_step_name} 성공.")
            node_return_output = {
                "final_response": response_text,
                "error_message": None,
                "current_step": current_step_name
            }

    save_step_output(
        run_id=run_id,
        step_name=current_step_name,
        step_inputs=step_inputs_for_saving,
        step_outputs=node_return_output,
        status=status_for_saving,
        error_message=error_for_saving
    )
    return node_return_output


workflow = StateGraph(GraphState)
workflow.add_node("extract_image_info", node_extract_image_info)
# 🆕 사용자 의도 분석 에이전트 노드 추가
workflow.add_node("refine_user_intent", node_refine_user_intent)
workflow.add_node("enrich_product_info", node_enrich_product_info)
workflow.add_node("evaluate_product", node_evaluate_product)
workflow.add_node("generate_response", node_generate_natural_response)

workflow.set_entry_point("extract_image_info")

workflow.add_edge("extract_image_info", "refine_user_intent")
workflow.add_edge("refine_user_intent", "enrich_product_info")
workflow.add_edge("enrich_product_info", "evaluate_product")
workflow.add_edge("evaluate_product", "generate_response")
workflow.add_edge("generate_response", END)
app = workflow.compile()


if __name__ == "__main__":
    TEST_IMAGE_DIR = "img"
    test_image_filename = "propolis_1.png"
    test_image_path = os.path.join(TEST_IMAGE_DIR, test_image_filename)
    sample_user_query = "입에 뭐 났는데 이거 먹으면 효과 있나요?"

    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"🚨 오류: 이미지 디렉터리 '{TEST_IMAGE_DIR}'를 찾을 수 없습니다.")
        print("디렉토리를 생성하고 테스트 이미지를 넣어주세요.")
    elif not os.path.exists(test_image_path):
        print(f"🚨 오류: 테스트 이미지 '{test_image_path}'를 찾을 수 없습니다.")
        print(f"'{TEST_IMAGE_DIR}' 폴더에 '{test_image_filename}' 파일을 넣어주세요.")
    else:
        print("🚀 LangGraph 기반 통합 파이프라인 시작 🚀")
        # Run ID 생성 및 파이프라인 시작 시간 기록
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        pipeline_start_time = datetime.now()
        print(f"✨ 실행 ID: {run_id}")

        initial_state = {
            "image_path": test_image_path,
            "user_query": sample_user_query,
            "run_id": run_id, # 초기 상태에 run_id 추가
            # 초기에는 다른 필드들은 None 또는 빈 값으로 시작
            "image_data": None,
            "product_name_from_image": None,
            "enriched_info": None,
            "refined_user_query": None,
            "is_query_ambiguous": None,
            "evaluation_result": None,
            "data_consistency_status": None,
            "validation_issues": None,
            "final_response": None,
            "current_step": None,
            "error_message": None
        }
    
        final_state = app.invoke(initial_state)
        pipeline_end_time = datetime.now() # ◀️ 파이프라인 종료 시간 기록

        # 🔽 실행 메타데이터 저장
        save_run_metadata(run_id, initial_state, final_state, pipeline_start_time, pipeline_end_time)

        print("\n--- 📈 최종 그래프 상태 ---")
        # 최종 상태 출력 시 run_id를 포함한 주요 정보만 간략히 출력하거나, 전체를 출력할 수 있습니다.
        # 여기서는 이전처럼 전체를 출력합니다.
        print(json.dumps(final_state, indent=2, ensure_ascii=False))
        print("-" * 50)


        if final_state.get("error_message"):
            print(f"\n🚫 [{run_id}] 파이프라인 중 오류 발생: {final_state['error_message']}")
            print(f"오류 발생 단계: {final_state.get('current_step', '알 수 없음')}")
        elif final_state.get("final_response"):
             print(f"\n🎉 [{run_id}] 파이프라인 성공적으로 완료! 🎉")
             print("\n---  최종 생성된 답변 ---")
             print(final_state["final_response"])
        elif final_state.get("error_message") and final_state.get("current_step") == "generate_natural_response" and not final_state.get("final_response"):
            print(f"\n⚠️ [{run_id}] 파이프라인은 완료되었으나, 최종 답변 생성 중 오류 발생: {final_state['error_message']}")
            print(f"오류 발생 단계: {final_state.get('current_step', '알 수 없음')}")
        else: # 오류는 없으나 최종 답변이 없는 경우 등
            print(
                f"\n🤔 [{run_id}] 파이프라인은 완료되었지만, 최종 답변이 없거나 알 수 없는 문제가 발생했습니다. 최종 상태를 확인해주세요."
            )
