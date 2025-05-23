from typing import TypedDict, Dict, Any, Optional, List

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
    data_consistency_status: Optional[str] # 예: "일치함", "부분적 불일치", "주요 불일치