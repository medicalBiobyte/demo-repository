import os
import json

# from dotenv import load_dotenv # main.py에서 처리
from core.prompt import WEB2INGREDIENT_PROMPT  # 아래에서 만들 프롬프트
from core.config import web_search_llm, text_llm  # TavilyClient, ChatOpenAI
from typing import List
import re

# 📁 설정
RESULT_ALL_PATH = "IMG2TEXT_data/result_all.json"
OUTPUT_DIR = "TEXT2SEARCH_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 🔍 검색 → 요약 텍스트 추출
def search_product_and_summarize(product_name: str) -> str:
    search_query = f"{product_name} 성분 효능"
    print(f"🌐 '{search_query}'로 웹 검색 중...")
    search_result = web_search_llm.search(search_query)
    results = search_result.get("results", [])

    processed_content_parts = []
    if not results:
        print(f"⚠️ '{product_name}'에 대한 웹 검색 결과가 없습니다.")
        return ""

    for res in results:
        title = res.get("title", "제목 없음")
        snippet = res.get("content", "내용 없음")
        source_url = res.get("url")  # 웹 검색 결과에서 URL 추출

        # 각 결과를 "제목 - 내용 - 출처 URL" 형식으로 구성
        part = f"[{title}]\n{snippet}"
        if source_url:
            part += f"\n출처: {source_url}"
        else:
            part += "\n출처: 정보 없음"  # URL이 없는 경우를 대비
        processed_content_parts.append(part)

    # 모든 검색 결과를 두 줄 바꿈으로 연결하여 하나의 텍스트로 만듭니다.
    # 이는 WEB2INGREDIENT_PROMPT에서 "각 문단 말미: 해당 정보의 출처 URL" 형식을 따르도록 합니다.
    return "\n\n".join(processed_content_parts).strip()


# 🧠 LLM을 통해 성분 + 효능 추출
def extract_ingredients_and_effects(summary_text: str) -> dict:
    if not summary_text.strip():  # summary_text가 비어있거나 공백만 있는 경우
        print("⚠️ 요약 텍스트가 비어 있어 성분 및 효능 추출을 건너뜁니다.")
        return {}

    full_prompt = WEB2INGREDIENT_PROMPT.replace("{web_text}", summary_text)

    response = text_llm.invoke(full_prompt)
    raw_text = response.content

    def extract_json_string(text: str) -> str:
        match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    cleaned = extract_json_string(raw_text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"❌ JSON 파싱 실패. 원본 응답:\n---\n{raw_text}\n---")
        # 파싱 실패 시, 추가적인 디버깅 정보나 빈 결과를 반환할 수 있습니다.
        return {"error": "JSON 파싱 실패", "raw_response": raw_text}


# --- 파이프라인을 위한 새로운 함수 ---
def get_enriched_product_info(product_name: str) -> dict:
    if not product_name:
        print("⚠️ 제품명이 제공되지 않았습니다. (get_enriched_product_info)")
        return {"제품명": product_name, "error": "제품명 없음"}

    print(f"🔍 '{product_name}'에 대한 웹 검색 및 성분 추출 중...")
    web_summary = search_product_and_summarize(product_name)
    if not web_summary:
        print(f"⚠️ '{product_name}'에 대한 웹 요약을 가져올 수 없습니다.")
        return {"제품명": product_name, "error": "웹 요약 실패", "요약_텍스트": ""}

    parsed_result = extract_ingredients_and_effects(web_summary)

    # LLM 결과에 제품명과 원본 요약 텍스트 추가
    # parsed_result가 에러 객체일 수도 있으므로, 안전하게 업데이트
    if isinstance(parsed_result, dict):
        parsed_result["제품명"] = product_name
        parsed_result["요약_텍스트"] = web_summary
    else:  # LLM 결과가 예상치 못한 형식일 경우 (예: 파싱 완전 실패로 문자열 반환 등)
        parsed_result = {
            "제품명": product_name,
            "error": "LLM 결과 처리 실패",
            "요약_텍스트": web_summary,
            "llm_raw_output": parsed_result,  # 원본 LLM 출력을 저장
        }

    if "error" not in parsed_result:
        print(f"✅ '{product_name}' 정보 보강 완료.")
    else:
        # 이미 parsed_result에 에러 정보가 있을 수 있음 (JSON 파싱 실패 등)
        print(
            f"⚠️ '{product_name}' 정보 보강 중 문제 발생: {parsed_result.get('error', '알 수 없는 오류')}"
        )
        # 다음 단계를 위해 최소한의 정보와 오류를 포함하여 반환 (확정_성분은 없을 수 있으므로 기본값 제공)
        if "확정_성분" not in parsed_result:
            parsed_result["확정_성분"] = []

    return parsed_result


# 🚀 전체 파이프라인 실행
# def process_all_products():
#     with open(RESULT_ALL_PATH, encoding="utf-8") as f:
#         products = json.load(f)

#     for entry in products:
#         product_name = entry.get("제품명", "").split("/")[0].strip()
#         if not product_name:
#             continue

#         print(f"🔍 웹 검색 및 성분 추출 중: {product_name}")
#         result_path = os.path.join(OUTPUT_DIR, f"enriched_{product_name}.json")

#         if os.path.exists(result_path):
#             print(f"✅ 이미 처리됨: {product_name}")
#             continue

#         # 1. 검색 + 요약
#         web_summary = search_product_and_summarize(product_name)

#         # 2. 요약문 + 프롬프트 → LLM 처리
#         parsed_result = extract_ingredients_and_effects(web_summary)

#         if parsed_result:
#             parsed_result["제품명"] = product_name
#             parsed_result["요약_텍스트"] = web_summary  # Optional: 출처 기록용

#             with open(result_path, "w", encoding="utf-8") as f:
#                 json.dump(parsed_result, f, ensure_ascii=False, indent=2)

#             print(f"✅ 저장 완료: {result_path}")
#         else:
#             print(f"⚠️ 실패: {product_name}")


if __name__ == "__main__":
    test_product_name = "키즈픽션"
    enriched_info = get_enriched_product_info(test_product_name)

    # ✅ 결과 저장 추가
    if enriched_info:
        safe_name = test_product_name.replace(" ", "_").replace("/", "_")
        result_path = os.path.join(OUTPUT_DIR, f"enriched_{safe_name}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(enriched_info, f, ensure_ascii=False, indent=2)
        print(f"✅ 결과 저장 완료: {result_path}")

    # 기존의 process_all_products()를 사용하려면 아래 주석을 해제하세요.
    # process_all_products()
