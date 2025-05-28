import pandas as pd
from konlpy.tag import Okt
from collections import Counter
import re

fnclty_path = "csv_data/fnclty_materials_complete.csv"
drug_path = "csv_data/drug_raw.csv"
healthfood_claims_path = "csv_data/healthfood_claims_final10.csv"

df_healthfood = pd.read_csv(healthfood_claims_path)

# --- 텍스트 데이터 전처리 및 카테고리 분류 함수 정의 ---
okt = Okt()
stopwords = ['에', '은', '는', '이', '가', '을', '를', '의', '들', '좀', '잘', '걍', '과', '도', '으로', '자', '와', '한', '하다', '되다', '있다', '없다', '이다', '좋다', '많다', '그리고', '하지만', '그러나', '그래서', '또는', '및', '등', '수', '것', '줄', '알', '더', '그', '저', '이런', '저런', '그런', '위해', '때문', '통해', '대한', '관련', '도움', '기능성']

def preprocess_text(text):
    if pd.isna(text) or text == '': # 결측치 또는 빈 문자열 처리
        return []
    text_cleaned = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', str(text)) # 한글 및 공백만 남기기
    tokens = okt.pos(text_cleaned, stem=True, norm=True)
    words = [word for word, tag in tokens if tag in ['Noun', 'Adjective', 'Verb'] and word not in stopwords and len(word) > 1] # 명사, 형용사, 동사 중 2글자 이상
    return words

category_keywords_updated = {
    "혈당·콜레스테롤 조절": ["혈당", "콜레스테롤", "식후혈당", "LDL", "인슐린", "중성지질", "당뇨", "고지혈증"],
    "장건강·소화": ["장", "장내", "배변", "소화", "변비", "유산균", "프로바이오틱스", "프리바이오틱스", "식이섬유", "정장"],
    "면역기능강화": ["면역", "면역력", "면역세포", "면역과민", "항산화", "아연", "비타민C", "베타글루칸", "프로폴리스", "알로에"],
    "항산화·세포건강": ["항산화", "유해산소", "세포보호", "세포노화", "피부건강", "코엔자임Q10", "비타민E", "셀레늄", "폴리페놀"],
    "뼈·관절·치아건강": ["뼈", "골밀도", "칼슘", "관절", "연골", "비타민D", "마그네슘", "MSM", "NAG", "글루코사민", "치아"],
    "기억력·인지기능개선": ["기억력", "인지", "집중력", "뇌건강", "뇌기능", "오메가3", "DHA", "EPA", "은행잎", "포스파티딜세린", "테아닌"],
    "혈행·혈압건강": ["혈행", "혈압", "혈관", "혈액순환", "호모시스테인", "오메가3", "EPA", "홍국", "코큐텐", "폴리코사놀", "나토키나제"],
    "피부건강·미용": ["피부", "보습", "자외선", "탄력", "주름", "모발", "콜라겐", "히알루론산", "세라마이드", "비오틴"],
    "간건강": ["간", "간기능", "피로", "숙취", "밀크씨슬", "실리마린", "헛개", "UDCA"],
    "체지방감소·에너지생성": ["체지방", "다이어트", "체중", "기초대사량", "에너지", "피로개선", "지구력", "활력", "가르시니아", "HCA", "녹차", "카테킨", "L카르니틴", "CLA", "비타민B"],
    "눈건강": ["눈", "시력", "피로도", "황반", "루테인", "지아잔틴", "아스타잔틴", "빌베리", "오메가3", "DHA"],
    "스트레스완화·수면질개선": ["스트레스", "긴장완화", "수면", "수면질", "심신안정", "테아닌", "홍경천", "락티움", "감태"]
}

def assign_category(text_tokens):
    if not text_tokens:
        return "분류불가"
    category_scores = {category: 0 for category in category_keywords_updated.keys()}
    for token in text_tokens:
        for category, keywords in category_keywords_updated.items():
            for keyword in keywords:
                if keyword in token or token in keyword: # 키워드가 토큰을 포함하거나, 토큰이 키워드를 포함하는 경우 (더 유연하게)
                    category_scores[category] += 1
    if sum(category_scores.values()) == 0:
        return "분류불가"
    best_category = max(category_scores, key=category_scores.get)
    if category_scores[best_category] == 0 : # 하나도 매칭 안되면 분류불가
        return "분류불가"
    return best_category

df_['processed_text'] = df_['기능성 내용'].apply(preprocess_text)
df_['category'] = df_['processed_text'].apply(assign_category)

print("### 카테고리 분류 결과 (상위 5개) ###")
print(df_[['기능성 내용', 'processed_text', 'category']].head())
print("\n" + "="*50 + "\n")

print("### 카테고리별 데이터 수 ###")
print(df_['category'].value_counts())
print("\n" + "="*50 + "\n")


# --- 3단계: 범주별 키워드 분석 (단어 빈도 분석) ---
def analyze_keywords_by_category(df_input):
    if 'category' not in df_input.columns or 'processed_text' not in df_input.columns:
        print("오류: 키워드 분석에 필요한 'category' 또는 'processed_text' 컬럼이 없습니다.")
        return None
    df_analyzable = df_input[df_input['category'] != "분류불가"].copy()
    if df_analyzable.empty:
        print("분석 가능한 데이터가 없습니다 ('분류불가' 카테고리만 존재하거나 데이터가 없음).")
        return None

    category_all_words = {}
    for category_name in df_analyzable['category'].unique():
        words_in_category = []
        # Series의 각 리스트 요소들을 extend로 합침
        df_analyzable[df_analyzable['category'] == category_name]['processed_text'].apply(lambda x: words_in_category.extend(x))
        category_all_words[category_name] = words_in_category

    category_word_counts = {}
    for category_name, words in category_all_words.items():
        if words:
            counts = Counter(words)
            category_word_counts[category_name] = counts.most_common(15) # 상위 15개 키워드
        else:
            category_word_counts[category_name] = []
    return category_word_counts

keyword_analysis_results = analyze_keywords_by_category(df_.copy())

if keyword_analysis_results:
    print("### 카테고리별 상위 키워드 분석 결과 ###")
    for category_name, keywords in keyword_analysis_results.items():
        print(f"\n--- {category_name} ---")
        if keywords:
            # 키워드를 보기 좋게 출력
            keyword_str_list = [f"{word}({count})" for word, count in keywords]
            print(", ".join(keyword_str_list))
        else:
            print("추출된 주요 키워드가 없습니다.")
    print("\n" + "="*50 + "\n")
else:
    print("키워드 분석 결과를 생성하지 못했습니다.")