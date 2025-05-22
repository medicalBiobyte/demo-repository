import os
import json
import pandas as pd
# from dotenv import load_dotenv # main.py에서 처리
from .config import text_llm
from .prompt import QUERY2KEYWORD_PROMPT
from typing import List

# 📁 경로 설정

# claim_check_3.py 파일의 현재 디렉터리 위치를 가져옵니다.
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # core 폴더의 부모 디렉터리 (프로젝트 최상위)
CSV_DATA_DIR = os.path.join(BASE_DIR, "csv_data")

# DATA_DIR = "TEXT2SEARCH_data"
CSV_FNCLTY = os.path.join(CSV_DATA_DIR, "fnclty_materials_complete.csv")
CSV_DRUG = os.path.join(CSV_DATA_DIR, "drug_raw.csv")
# OUTPUT_DIR = "DECISION_data" # main.py에서 처리
# os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# --- 파이프라인을 위한 수정된 함수 ---
def get_product_evaluation(enriched_data: dict, user_query: str) -> dict:
    product_name = enriched_data.get("제품명", "unknown").strip()
    # web_search_2.py의 get_enriched_product_info 반환값에서 "확정_성분" 키를 사용
    ingredients = enriched_data.get("확정_성분", [])

    if not isinstance(ingredients, list): # ingredients가 리스트가 아닐 경우 처리
        print(f"⚠️ '{product_name}'의 확정_성분 형식이 잘못되었습니다: {ingredients}. 빈 리스트로 처리합니다.")
        ingredients = []

    query_keywords = extract_keywords_from_query(user_query)

    matched_results = []
    match_count = 0

    for ing in ingredients:
        efficacy = efficacy_dict.get(ing)
        if not efficacy: # efficacy_dict에 성분 정보가 없는 경우
            matched_results.append({"성분명": ing, "효능": "정보 없음", "일치도": "정보 없음"})
            continue

        match_level = match_efficacy(query_keywords, efficacy)
        if match_level == "일치":
            match_count += 1
        matched_results.append({"성분명": ing, "효능": efficacy, "일치도": match_level})

    fallback_result = {}
    # 성분 기반 일치가 없거나, 애초에 성분 정보가 없었던 경우 제품명 기반 보완 시도
    if match_count == 0:
        if product_name in drug_efficacy_dict:
            fallback_text = drug_efficacy_dict[product_name]
            fallback_match = match_efficacy(query_keywords, fallback_text)
            fallback_result = {
                "제품명": product_name, # 이전에 누락되었던 제품명 추가
                "보완_효능": fallback_text,
                "일치도": fallback_match,
            }
            if fallback_match == "일치":
                match_count += 1 # 전체적인 일치 카운트 증가
        else:
            if not ingredients:
                 print(f"ℹ️ '{product_name}'에 대한 성분 정보가 없고, 제품명 기반 보완 정보도 없습니다.")
            else: # ingredients는 있었지만 매칭이 안 된 경우
                 print(f"ℹ️ '{product_name}' 성분 기반 일치 항목이 없고, 제품명 기반 보완 정보도 없습니다.")


    if match_count >= 1:
        final_judgement_text = "사용자 질문과 일부 성분 또는 제품의 효능이 일치합니다."
    else:
        final_judgement_text = "광고 주장의 근거가 부족합니다 (불일치)"

    evaluation_output = {
        "제품명": product_name,
        "사용자_질문": user_query,
        "질문_핵심_키워드": query_keywords,
        "확정_성분": ingredients, # 입력으로 받은 확정_성분
        "매칭_성분": matched_results,
        "제품명_기반_보완": fallback_result,
        "최종_판단": final_judgement_text,
        # 다음 단계(자연어 답변 생성)에서 풍부한 정보를 활용할 수 있도록 추가 데이터 포함
        "original_효능_주장": enriched_data.get("original_효능_주장"), # 1단계의 이미지 추출 효능 주장
        "web_요약": enriched_data.get("요약_텍스트"), # 2단계의 웹 검색 요약
        "성분_추출_출처": enriched_data.get("성분_추출_출처"), # 2단계의 성분 출처
        "성분_효능_웹": enriched_data.get("성분_효능") # 2단계의 웹 기반 성분 효능
    }

    print(f"✅ '{product_name}' 평가 완료: {final_judgement_text}")
    # 파일 저장 로직은 main.py에서 필요시 수행
    return evaluation_output



# 🧪 제품 평가 함수
# def evaluate_product(file_path: str, user_query: str):
#     with open(file_path, encoding="utf-8") as f:
#         product_data = json.load(f)

#     product_name = product_data.get("제품명", "unknown").strip()
#     ingredients = product_data.get("확정_성분", [])

#     query_keywords = extract_keywords_from_query(user_query)

#     matched_results = []
#     match_count = 0

#     for ing in ingredients:
#         efficacy = efficacy_dict.get(ing)
#         if not efficacy:
#             matched_results.append(
#                 {"성분명": ing, "효능": "없음", "일치도": "정보 없음"}
#             )
#             continue

#         match_level = match_efficacy(query_keywords, efficacy)
#         if match_level == "일치":
#             match_count += 1

#         matched_results.append({"성분명": ing, "효능": efficacy, "일치도": match_level})

#     # 🎯 성분 일치 결과가 없는 경우 → 제품명 기반 보완
#     fallback_result = {}
#     if match_count == 0 and product_name in drug_efficacy_dict:
#         fallback_text = drug_efficacy_dict[product_name]
#         fallback_match = match_efficacy(query_keywords, fallback_text)
#         fallback_result = {
#             "제품명": product_name,
#             "보완_효능": fallback_text,
#             "일치도": fallback_match,
#         }
#         if fallback_match == "일치":
#             match_count += 1

#     # 종합 판단
#     if match_count >= 1:
#         result = "사용자 질문과 일부 성분 또는 제품의 효능이 일치합니다."
#     else:
#         result = "광고 주장의 근거가 부족합니다 (불일치)"

#     final = {
#         "제품명": product_name,
#         "사용자_질문": user_query,
#         "질문_핵심_키워드": query_keywords,
#         "확정_성분": ingredients,
#         "매칭_성분": matched_results,
#         "제품명_기반_보완": fallback_result,
#         "최종_판단": result,
#     }

#     filename = f"enriched_{product_name}.json"
#     with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
#         json.dump(final, f, ensure_ascii=False, indent=2)

#     print(f"✅ 평가 완료: {product_name} → {result}")


# # ▶️ 실행 예시
# if __name__ == "__main__":
#     query = "이 약은 키 크는데 도움이 되나요?"

#     for filename in os.listdir(DATA_DIR):
#         if filename.endswith(".json"):
#             filepath = os.path.join(DATA_DIR, filename)
#             evaluate_product(filepath, query)
