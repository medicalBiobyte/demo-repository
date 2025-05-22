import os
import json
# from dotenv import load_dotenv # main.py에서 처리
from .prompt import WEB2INGREDIENT_PROMPT  # 아래에서 만들 프롬프트
from .config import web_search_llm, text_llm  # TavilyClient, ChatOpenAI
from typing import List
import re

# 📁 설정
RESULT_ALL_PATH = "IMG2TEXT_data/result_all.json"
OUTPUT_DIR = "TEXT2SEARCH_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 🔍 검색 → 요약 텍스트 추출
def search_product_and_summarize(product_name: str) -> str:
    search_result = web_search_llm.search(product_name + " 성분")  # or "성분 효능"
    results = search_result.get("results", [])

    content = ""
    for res in results:
        title = res.get("title", "")
        snippet = res.get("content", "")
        content += f"[{title}]\n{snippet}\n\n"

    return content.strip()


# 🧠 LLM을 통해 성분 + 효능 추출
def extract_ingredients_and_effects(summary_text: str) -> dict:
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
        print("❌ JSON 파싱 실패\n", raw_text)
        return {}
    
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

    if parsed_result: # 파싱 성공 시 (내용이 비어있을 순 있음)
        parsed_result["제품명"] = product_name # 제품명 정보 추가
        parsed_result["요약_텍스트"] = web_summary # 요약 텍스트도 결과에 포함 (answer_user_4.py에서 활용 가능)
        print(f"✅ '{product_name}' 정보 보강 완료.")
        return parsed_result
    else: # LLM 파싱 실패 시
        print(f"⚠️ '{product_name}' 정보 보강 실패 (LLM 파싱).")
        # 다음 단계를 위해 최소한의 정보와 오류를 포함하여 반환
        return {"제품명": product_name, "error": "LLM 파싱 실패", "요약_텍스트": web_summary, "확정_성분": []}


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
    process_all_products()
