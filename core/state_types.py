from typing import TypedDict, Dict, Any, Optional, List

class GraphState(TypedDict):
    image_path: str
    user_query: str
    run_id: Optional[str] # 각 실행을 식별하는 ID 추가
    archived_image_path: Optional[str] # 아카이브된 원본 이미지 경로 필드 추가
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