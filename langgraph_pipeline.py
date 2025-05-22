import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
# 기존 core 모듈 임포트
from core.text_extract_1 import extract_info_from_image
from core.web_search_2 import get_enriched_product_info
from core.claim_check_3 import get_product_evaluation
from core.answer_user_4 import generate_natural_response

load_dotenv()

# --- 1. 상태 정의 (GraphState) ---
# 각 노드 간에 전달될 데이터의 구조를 정의합니다.
class GraphState(TypedDict):
    image_path: str               # 입력: 테스트할 이미지 경로
    user_query: str               # 입력: 사용자 질문
    
    # 중간 결과 및 최종 결과
    image_data: Optional[Dict[str, Any]]        # 1단계: 이미지 추출 정보
    product_name_from_image: Optional[str]      # 1단계: 이미지에서 추출된 제품명
    enriched_info: Optional[Dict[str, Any]]     # 2단계: 웹 정보 보강 결과
    evaluation_result: Optional[Dict[str, Any]] # 3단계: 제품 평가 결과
    final_response: Optional[str]               # 4단계: 최종 자연어 답변
    
    # 오류 처리 및 진행 상태
    error_message: Optional[str]  # 오류 발생 시 메시지 저장
    current_step: Optional[str]   # 현재 실행 중인 단계 표시용

# --- 2. 노드 함수 정의 ---
# 각 노드는 GraphState를 입력으로 받고, 변경된 상태 부분을 담은 딕셔너리를 반환합니다.

def node_extract_image_info(state: GraphState) -> Dict[str, Any]:
    """1단계: 이미지에서 정보 추출"""
    state["current_step"] = "extract_image_info"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")
    image_path = state["image_path"]

    if not os.path.exists(image_path):
        error_msg = f"이미지 파일을 찾을 수 없습니다: {image_path}"
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    image_data_output = extract_info_from_image(image_path) #

    if not image_data_output or "제품명" not in image_data_output or not image_data_output.get("제품명"): #
        error_msg = "이미지에서 유효한 정보를 추출하지 못했습니다." #
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        if image_data_output:
             print(f"부분 데이터: {json.dumps(image_data_output, indent=2, ensure_ascii=False)}")
        return {"image_data": image_data_output, "error_message": error_msg}

    product_name = image_data_output.get("제품명", "").split("/")[0].strip() #
    if not product_name:
        error_msg = "추출된 제품명이 유효하지 않습니다." #
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"image_data": image_data_output, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공. 제품명: {product_name}")
    return {
        "image_data": image_data_output,
        "product_name_from_image": product_name,
        "error_message": None 
    }

def node_enrich_product_info(state: GraphState) -> Dict[str, Any]:
    """2단계: 웹 검색으로 제품 정보 보강"""
    if state.get("error_message"): # 이전 단계에서 오류 발생 시 건너뛰기
        return {}
    state["current_step"] = "enrich_product_info"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    product_name = state["product_name_from_image"]
    if not product_name: # 이전 노드에서 처리되었어야 함
        error_msg = "정보 보강을 위한 제품명이 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    enriched_data = get_enriched_product_info(product_name) #

    if not enriched_data or enriched_data.get("error"): #
        error_msg = f"웹 정보를 보강하지 못했습니다. 메시지: {enriched_data.get('error', '알 수 없음') if enriched_data else '데이터 없음'}" #
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"enriched_info": enriched_data, "error_message": error_msg}

    # 원본 스크립트처럼 image_data의 '효능_주장'을 enriched_info에 추가
    if state.get("image_data"):
        enriched_data["original_효능_주장"] = state["image_data"].get("효능_주장") #

    print(f"✅ {state['current_step']} 성공.")
    return {"enriched_info": enriched_data, "error_message": None}


def node_evaluate_product(state: GraphState) -> Dict[str, Any]:
    """3단계: 사용자 질문 기반 제품 평가"""
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

    evaluation_data = get_product_evaluation(enriched_info, user_query) #

    if not evaluation_data or "최종_판단" not in evaluation_data: #
        error_msg = "제품 평가에 실패했습니다." #
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        if evaluation_data:
            print(f"부분 데이터: {json.dumps(evaluation_data, indent=2, ensure_ascii=False)}")
        return {"evaluation_result": evaluation_data, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공.")
    return {"evaluation_result": evaluation_data, "error_message": None}

def node_generate_natural_response(state: GraphState) -> Dict[str, Any]:
    """4단계: 자연어 답변 생성"""
    if state.get("error_message"):
        return {}
    state["current_step"] = "generate_natural_response"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    evaluation_result = state["evaluation_result"]
    if not evaluation_result:
        error_msg = "자연어 답변 생성을 위한 평가 결과가 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    response_text = generate_natural_response(evaluation_result) #

    if not response_text or not isinstance(response_text, str): #
        error_msg = f"자연어 답변 생성에 실패했거나 결과가 문자열이 아닙니다. (타입: {type(response_text)})" #
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"final_response": response_text, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공.")
    return {"final_response": response_text, "error_message": None}

# --- 3. 그래프 구성 ---
# StateGraph 객체를 생성하고 GraphState를 상태로 지정합니다.
workflow = StateGraph(GraphState)

# 노드 추가: 각 노드에 이름과 실행할 함수를 지정합니다.
workflow.add_node("extract_image_info", node_extract_image_info)
workflow.add_node("enrich_product_info", node_enrich_product_info)
workflow.add_node("evaluate_product", node_evaluate_product)
workflow.add_node("generate_response", node_generate_natural_response)

# --- 4. 엣지(연결) 정의 ---
# 노드 간의 실행 순서를 정의합니다.

# 진입점(Entry Point) 설정
workflow.set_entry_point("extract_image_info")

# 순차적 연결
workflow.add_edge("extract_image_info", "enrich_product_info")
workflow.add_edge("enrich_product_info", "evaluate_product")
workflow.add_edge("evaluate_product", "generate_response")
workflow.add_edge("generate_response", END) 

# --- 5. 그래프 컴파일 ---

# 정의된 워크플로우를 실행 가능한 객체로 컴파일합니다.
app = workflow.compile()

# --- 6. 그래프 실행 (예시) ---
if __name__ == "__main__":
    # 테스트용 입력값 설정 
    TEST_IMAGE_DIR = "img"  # 실제 이미지 폴더 경로
    test_image_filename = "height_medi_1.png"  # 테스트할 이미지 파일명
    test_image_path = os.path.join(TEST_IMAGE_DIR, test_image_filename)
    sample_user_query = "이거 먹으면 키 크는데 효과 있나요?" #

    # 실행 전 이미지 파일 존재 여부 확인
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"🚨 오류: 이미지 디렉토리 '{TEST_IMAGE_DIR}'를 찾을 수 없습니다.")
        print("디렉토리를 생성하고 테스트 이미지를 넣어주세요.")
    elif not os.path.exists(test_image_path): #
        print(f"🚨 오류: 테스트 이미지 '{test_image_path}'를 찾을 수 없습니다.") #
        print(f"'{TEST_IMAGE_DIR}' 폴더에 '{test_image_filename}' 파일을 넣어주세요.")
    else:
        print("🚀 LangGraph 기반 통합 파이프라인 시작 🚀")
        
        initial_state = {
            "image_path": test_image_path,
            "user_query": sample_user_query
        }

        # 그래프 실행 방법 1: 스트림으로 중간 결과 확인
        # print("\n--- 📢 스트림 이벤트 ---")
        # for event_chunk in app.stream(initial_state):
        #     for node_name, output_data in event_chunk.items():
        #         print(f"Node '{node_name}' 출력:")
        #         print(json.dumps(output_data, indent=2, ensure_ascii=False))
        #         print("-" * 30)

        # 그래프 실행 방법 2: invoke로 최종 상태 확인
        final_state = app.invoke(initial_state)

        print("\n--- 📈 최종 그래프 상태 ---")
        print(json.dumps(final_state, indent=2, ensure_ascii=False))
        print("-" * 50)

        if final_state.get("error_message"):
            print(f"\n🚫 파이프라인 중 오류 발생: {final_state['error_message']}")
            print(f"오류 발생 단계: {final_state.get('current_step', '알 수 없음')}")
        elif final_state.get("final_response"):
            print("\n🎉 파이프라인 성공적으로 완료! 🎉")
            print("\n---  최종 생성된 자연어 답변 ---") #
            print(final_state["final_response"])
        else:
            print("\n🤔 파이프라인은 완료되었으나, 최종 답변이 없거나 알 수 없는 문제가 발생했습니다.")

