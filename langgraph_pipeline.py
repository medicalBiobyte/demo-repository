import os
import json
import re
from dotenv import load_dotenv
from typing import TypedDict, Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from datetime import datetime
from core.config import text_llm # text_llm 가져오기
from core.prompt import INTENT_REFINEMENT_PROMPT, DATA_VALIDATION_PROMPT # 새로운 프롬프트 가져오기

# 기존 core 모듈 임포트
from core.text_extract_1 import extract_info_from_image, extract_json_string
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

    # 사용자 의도 분석 에이전트 관련 필드
    is_query_ambiguous: Optional[bool]
    clarification_questions: Optional[List[str]]
    refined_user_query: Optional[str] # 정제된 사용자 질문 또는 핵심 의도
    original_query_keywords: Optional[List[str]] # 초기 키워드 (claim_check_3.py 에서 생성)

    # 데이터 검증 에이전트 관련 필드
    validation_issues: Optional[List[Dict[str, str]]] # 검증 시 발견된 문제점 목록
    data_consistency_status: Optional[str] # 예: "일치함", "부분적 불일치", "주요 불일치"

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
            print("\n---  최종 생성된 답변 ---")
            print(final_state["final_response"])
        else:
            print(
                "\n🤔 파이프라인은 완료되었지만, 최종 답변이 없거나 알 수 없는 문제가 발생했습니다."
            )
