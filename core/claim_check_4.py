import os
import json
import pandas as pd
# from dotenv import load_dotenv # main.py에서 처리
from .config import text_llm
from .prompt import QUERY2KEYWORD_PROMPT
from typing import List, Tuple
from rapidfuzz import fuzz   # pip install rapidfuzz

# 📁 경로 설정

# claim_check_3.py 파일의 현재 디렉터리 위치를 가져옵니다.
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # core 폴더의 부모 디렉터리 (프로젝트 최상위)
CSV_DATA_DIR = os.path.join(BASE_DIR, "csv_data")
CSV_HEALTHFOOD_CLAIMS = os.path.join(CSV_DATA_DIR, "healthfood_claims_final10.csv")

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

# 📄 CSV: (제품명, 성분명['일일섭취량']) 복합 키 기반 효능 (healthfood_claims_final10.csv) 
df_healthfood_claims = pd.read_csv(CSV_HEALTHFOOD_CLAIMS)

PRODUCT_NAME_COL_HC_CLAIMS = "제품명"
INGREDIENT_COL_HC_CLAIMS = "일일섭취량"
EFFICACY_COL_HC_CLAIMS = "기능성 내용"

healthfood_claims_composite_key_efficacy_dict: dict[Tuple[str, str], str] = {}

# healthfood_claims_final10.csv 파일에 필요한 컬럼들이 모두 있는지 확인
if PRODUCT_NAME_COL_HC_CLAIMS in df_healthfood_claims.columns and \
   INGREDIENT_COL_HC_CLAIMS in df_healthfood_claims.columns and \
   EFFICACY_COL_HC_CLAIMS in df_healthfood_claims.columns:
    
    df_healthfood_claims[PRODUCT_NAME_COL_HC_CLAIMS] = df_healthfood_claims[PRODUCT_NAME_COL_HC_CLAIMS].astype(str)
    df_healthfood_claims[INGREDIENT_COL_HC_CLAIMS] = df_healthfood_claims[INGREDIENT_COL_HC_CLAIMS].astype(str)
    df_healthfood_claims[EFFICACY_COL_HC_CLAIMS] = df_healthfood_claims[EFFICACY_COL_HC_CLAIMS].astype(str)

    for _, row in df_healthfood_claims.iterrows():
        product_name_key = row[PRODUCT_NAME_COL_HC_CLAIMS].strip()
        ingredient_key = row[INGREDIENT_COL_HC_CLAIMS].strip() # '일일섭취량' 컬럼 값 (성분명으로 사용)
        efficacy_value = row[EFFICACY_COL_HC_CLAIMS]    # '기능성 내용' 컬럼 값

        if product_name_key and ingredient_key: # 제품명과 성분명(일일섭취량)이 모두 유효한 경우
            healthfood_claims_composite_key_efficacy_dict[(product_name_key, ingredient_key)] = efficacy_value
else:
    print(f"⚠️ '{os.path.basename(CSV_HEALTHFOOD_CLAIMS)}' 파일에서 복합 키 생성에 필요한 컬럼('{PRODUCT_NAME_COL_HC_CLAIMS}', '{INGREDIENT_COL_HC_CLAIMS}', 또는 '{EFFICACY_COL_HC_CLAIMS}')을(를) 찾을 수 없습니다. 해당 딕셔너리가 비어있을 수 있습니다.")


# 🔍 LLM으로 질의 핵심어 추출
def extract_keywords_from_query(query: str) -> List[str]:
    prompt = QUERY2KEYWORD_PROMPT.replace("{query}", query)
    response = text_llm.invoke(prompt)
    try:
        content = response.content.strip() # strip()추가
        return json.loads(content)
    except Exception as e:
        print(f"❌ 키워드 JSON 파싱 실패: {e}\n원문: {response.content}")
        return []


def _normalize(text: str) -> str:
    """
    소문자 변환 + 공백·중점(·)·특수기호 제거
    """
    text = text.lower()
    text = re.sub(r"[\s·\.\,!?;:()\[\]{}]", "", text)
    return text

def match_efficacy(query_keywords: list[str],
                   efficacy_text: str,
                   threshold: int = 70) -> str:

    eff_norm = _normalize(efficacy_text)

    for kw in query_keywords:
        kw_norm = _normalize(kw)
        if fuzz.partial_ratio(kw_norm, eff_norm) >= threshold:
            return "일치"

    return "불일치"

# --- 파이프라인을 위한 수정된 함수 ---
def get_product_evaluation(enriched_data: dict, user_query: str, original_user_query_for_display: str) -> dict: # 새 인자 추가
    product_name_original = enriched_data.get("제품명", "unknown")
    product_name = product_name_original.strip() 
    ingredients = enriched_data.get("확정_성분", [])

    if not isinstance(ingredients, list): # ingredients가 리스트가 아닐 경우 처리
        print(f"⚠️ '{product_name}'의 확정_성분 형식이 잘못되었습니다: {ingredients}. 빈 리스트로 처리합니다.")
        ingredients = []
    
    # user_query는 내부 처리용 (정제된) 질문임
    query_keywords = extract_keywords_from_query(user_query)

    matched_results = []
    match_count = 0

    for ing_original in ingredients:
        ing = ing_original.strip()
        efficacy = None
        source_db = None
        
        # 1순위: efficacy_dict (fnclty_materials_complete.csv - 성분 기반)
        if ing in efficacy_dict:
            efficacy = efficacy_dict[ing]
            source_db = "fnclty_materials (ingredient)"
        # 2순위: healthfood_claims_composite_key_efficacy_dict (healthfood_claims_final10.csv - (제품명, 성분명['일일섭취량']) 복합 키 기반)
        elif (product_name, ing) in healthfood_claims_composite_key_efficacy_dict:
            efficacy = healthfood_claims_composite_key_efficacy_dict[(product_name, ing)]
            source_db = "healthfood_claims (product+ingredient)"
        
        if efficacy:
            match_level = match_efficacy(query_keywords, efficacy)
            if match_level == "일치":
                match_count += 1
            matched_results.append({"성분명": ing_original, "효능": efficacy, "일치도": match_level, "출처 DB": source_db})
        else:
            matched_results.append({"성분명": ing_original, "효능": "정보 없음 (성분/복합키 DB)", "일치도": "정보 없음", "출처 DB": "N/A"})

    fallback_result = {}
    # 성분 기반 또는 (제품+성분) 복합키 기반 일치가 없으면 제품명만으로 보완 시도
    if match_count == 0:
        # 제품명 기반 검색은 drug_efficacy_dict (drug_raw.csv)만 사용
        if product_name in drug_efficacy_dict:
            fallback_text = drug_efficacy_dict[product_name]
            fallback_match = match_efficacy(query_keywords, fallback_text)
            fallback_result = {
                "제품명": product_name_original,
                "보완_효능": fallback_text,
                "일치도": fallback_match,
                "출처 DB": "drug_raw (product)"
            }
            if fallback_match == "일치":
                match_count += 1
        else:
            sources_checked = ["fnclty_materials", "healthfood_claims_composite", "drug_raw"]
            if not ingredients:
                 print(f"ℹ️ '{product_name_original}'에 대한 성분 정보가 없고, 제품명 기반 보완 정보도 없습니다. (확인한 DB: {', '.join(sources_checked)})")
            else:
                 print(f"ℹ️ '{product_name_original}' 성분 및 (제품+성분)복합키 기반 일치 항목이 없고, 제품명 기반 보완 정보도 없습니다. (확인한 DB: {', '.join(sources_checked)})")

    if match_count >= 1:
        final_judgement_text = "사용자 질문과 일부 성분 또는 제품의 효능이 일치합니다."
    else:
        final_judgement_text = "광고 주장의 근거가 부족합니다 (불일치 또는 정보 없음)."

    evaluation_output = {
        "제품명": product_name_original,
        "사용자_질문": original_user_query_for_display,
        "내부_처리_질문": user_query,
        "질문_핵심_키워드": query_keywords,
        "확정_성분": ingredients,
        "매칭_성분": matched_results,
        "제품명_기반_보완": fallback_result,
        "최종_판단": final_judgement_text,
        "original_효능_주장": enriched_data.get("original_효능_주장"),
        "web_요약": enriched_data.get("요약_텍스트"),
        "성분_추출_출처": enriched_data.get("성분_추출_출처"),
        "성분_효능_웹": enriched_data.get("성분_효능")
    }

    print(f"✅ '{product_name_original}' 평가 완료 (원본 질문: '{original_user_query_for_display}', 내부 처리 질문: '{user_query}'): {final_judgement_text}")
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
