import os
import json
import pandas as pd
from dotenv import load_dotenv
from config import text_llm
from prompt import QUERY2KEYWORD_PROMPT
from typing import List

# 📁 경로 설정
DATA_DIR = "TEXT2SEARCH_data"
CSV_FNCLTY = "csv_data/fnclty_materials_complete.csv"
CSV_DRUG = "csv_data/drug_raw.csv"
OUTPUT_DIR = "DECISION_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📄 CSV: 성분 기반 효능
df_fnclty = pd.read_csv(CSV_FNCLTY)
df_fnclty["APLC_RAWMTRL_NM"] = df_fnclty["APLC_RAWMTRL_NM"].astype(str)
df_fnclty["FNCLTY_CN"] = df_fnclty["FNCLTY_CN"].astype(str)

efficacy_dict = {
    row["APLC_RAWMTRL_NM"]: row["FNCLTY_CN"] for _, row in df_fnclty.iterrows()
}

# 📄 CSV: 제품명 기반 효능 (보완용)
df_drug = pd.read_csv(CSV_DRUG)
df_drug["itemName"] = df_drug["itemName"].astype(str)
df_drug["efcyQesitm"] = df_drug["efcyQesitm"].astype(str)

drug_efficacy_dict = {
    row["itemName"]: row["efcyQesitm"] for _, row in df_drug.iterrows()
}


# 🔍 LLM으로 질의 핵심어 추출
def extract_keywords_from_query(query: str) -> List[str]:
    prompt = QUERY2KEYWORD_PROMPT.replace("{query}", query)
    response = text_llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except:
        print("❌ 키워드 추출 실패:\n", response.content)
        return []


# ✅ 키워드와 효능 텍스트 매칭
def match_efficacy(query_keywords: List[str], efficacy_text: str) -> str:
    text = efficacy_text.lower()
    matches = [kw for kw in query_keywords if kw in text]
    return "일치" if matches else "불일치"


# 🧪 제품 평가 함수
def evaluate_product(file_path: str, user_query: str):
    with open(file_path, encoding="utf-8") as f:
        product_data = json.load(f)

    product_name = product_data.get("제품명", "unknown").strip()
    ingredients = product_data.get("확정_성분", [])

    query_keywords = extract_keywords_from_query(user_query)

    matched_results = []
    match_count = 0

    for ing in ingredients:
        efficacy = efficacy_dict.get(ing)
        if not efficacy:
            matched_results.append(
                {"성분명": ing, "효능": "없음", "일치도": "정보 없음"}
            )
            continue

        match_level = match_efficacy(query_keywords, efficacy)
        if match_level == "일치":
            match_count += 1

        matched_results.append({"성분명": ing, "효능": efficacy, "일치도": match_level})

    # 🎯 성분 일치 결과가 없는 경우 → 제품명 기반 보완
    fallback_result = {}
    if match_count == 0 and product_name in drug_efficacy_dict:
        fallback_text = drug_efficacy_dict[product_name]
        fallback_match = match_efficacy(query_keywords, fallback_text)
        fallback_result = {
            "제품명": product_name,
            "보완_효능": fallback_text,
            "일치도": fallback_match,
        }
        if fallback_match == "일치":
            match_count += 1

    # 종합 판단
    if match_count >= 1:
        result = "사용자 질문과 일부 성분 또는 제품의 효능이 일치합니다."
    else:
        result = "광고 주장의 근거가 부족합니다 (불일치)"

    final = {
        "제품명": product_name,
        "사용자_질문": user_query,
        "질문_핵심_키워드": query_keywords,
        "확정_성분": ingredients,
        "매칭_성분": matched_results,
        "제품명_기반_보완": fallback_result,
        "최종_판단": result,
    }

    filename = f"enriched_{product_name}.json"
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"✅ 평가 완료: {product_name} → {result}")


# ▶️ 실행 예시
if __name__ == "__main__":
    query = "이 약은 키 크는데 도움이 되나요?"

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            evaluate_product(filepath, query)
