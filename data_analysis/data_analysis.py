import pandas as pd
from konlpy.tag import Okt
from collections import Counter
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# --- 파일 경로 설정 ---
fnclty_path = "csv_data/fnclty_materials_complete.csv"
drug_path = "csv_data/drug_raw.csv"
healthfood_claims_path = "csv_data/healthfood_claims_final10.csv"

# --- 데이터 로드 ---
df_healthfood = pd.read_csv(healthfood_claims_path)
df = df_healthfood.copy()

# --- 형태소 분석기 및 불용어 설정 ---
okt = Okt()
stopwords = [
    '에', '은', '는', '이', '가', '을', '를', '의', '들', '좀', '잘', '걍',
    '과', '도', '으로', '자', '와', '한', '하다', '되다', '있다', '없다',
    '이다', '좋다', '많다', '그리고', '하지만', '그러나', '그래서', '또는',
    '및', '등', '수', '것', '줄', '알', '더', '그', '저', '이런', '저런',
    '그런', '위해', '때문', '통해', '대한', '관련', '도움', '기능성'
]

# --- 텍스트 전처리 함수 ---
def preprocess_text(text):
    if pd.isna(text) or not text:
        return []
    text_clean = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', ' ', str(text))
    nouns = okt.nouns(text_clean)
    return [w for w in nouns if w not in stopwords and len(w) > 1]

# --- 카테고리 키워드 사전 (MFDS 기능성 영역 기반) ---
category_keywords = {
    "신경계": [
        "인지기능", "기억력", "집중력", "주의력", "학습능력",
        "스트레스 완화", "진정", "불안 개선", "기분 조절",
        "피로회복", "뇌피로", "멜라토닌", "가바"
    ],
    "소화/대사계": [
        "위 건강", "위장운동", "소화불량", "숙취해소",
        "장 건강", "배변활동", "변비", "설사 개선",
        "체지방 감소", "대사조절", "포만감", "식욕조절",
        "칼슘흡수", "칼슘대사", "비타민D",
        "프로바이오틱스", "프리바이오틱스"
    ],
    "생식&비뇨계": [
        "전립선 건강", "전립선비대증", "배뇨 기능", "요로 건강",
        "요로감염 예방", "성기능 개선", "남성호르몬 균형",
        "생리불순", "질 건강", "여성호르몬 균형"
    ],
    "신체방어·면역계": [
        "면역력 강화", "면역세포 증강", "NK세포 활성화",
        "항산화", "염증 완화", "항바이러스", "항균 작용",
        "인삼", "홍삼", "베타글루칸", "프로폴리스",
        "아연", "셀레늄"
    ],
    "감각계": [
        "눈 건강", "시력 보호", "황반변성 예방", "안구건조 완화",
        "청각 보호", "이명 완화", "치아 건강", "잇몸 강화",
        "구강 위생", "피부 건강", "보습", "탄력", "자외선 차단"
    ],
    "심혈관계": [
        "혈압 조절", "고혈압 완화", "혈관 탄력", "혈중 중성지방 개선",
        "LDL", "HDL 균형", "혈행 개선", "말초혈관 순환",
        "동맥경화 예방", "어지럼증 완화", "혈색소 관리",
        "나토키나제", "홍국", "코엔자임Q10"
    ],
    "내분비계": [
        "혈당 조절", "인슐린 저항성", "식후혈당 안정화",
        "갑상선 대사", "갑상선 기능 개선",
        "갱년기 증상 완화", "월경전증후군 개선", "생리통 완화",
        "블랙코호시", "보스웰리아"
    ],
    "근육계": [
        "뼈 건강", "관절 건강", "연골 보호", "관절염 완화",
        "근력 강화", "근지구력", "근육통 완화",
        "운동수행능력", "회복력", "근육피로 회복",
        "MSM", "글루코사민", "콘드로이틴",
        "콜라겐", "비타민C"
    ]
}

# --- 카테고리 분류 함수 ---
def assign_category_with_hits(tokens):
    if not tokens:
        return "분류불가", []
    norm_tokens = [t.lower().replace(" ", "") for t in tokens]
    hits = []
    matched_kws = []
    for category, kws in category_keywords.items():
        for kw in kws:
            norm_kw = kw.lower().replace(" ", "")
            for t in norm_tokens:
                if norm_kw in t or t in norm_kw:
                    hits.append(category)
                    matched_kws.append(kw)
    if not hits:
        return "분류불가", []
    best_cat = Counter(hits).most_common(1)[0][0]
    filtered_matches = [kw for kw in matched_kws if any(c == best_cat for c in hits)]
    return best_cat, list(set(filtered_matches))

# --- 분류 및 매칭된 키워드 컬럼 추가 ---
df[['category', 'matched_keywords']] = df['기능성 내용'].apply(preprocess_text).apply(
    lambda tokens: pd.Series(assign_category_with_hits(tokens))
)

# --- 결과 출력 ---
print("### 카테고리 분류 결과 (상위 5개) ###")
print(df[['기능성 내용', 'matched_keywords', 'category']].head(), "\n")
print("### 카테고리별 데이터 수 ###")
print(df['category'].value_counts(), "\n")

# --- 카테고리별 키워드 빈도 분석 함수 ---
def analyze_keywords_by_category(df_input):
    if 'category' not in df_input.columns or 'matched_keywords' not in df_input.columns:
        print("오류: 'category' 또는 'matched_keywords' 컬럼이 없습니다.")
        return None
    df_usable = df_input[df_input['category'] != "분류불가"]
    if df_usable.empty:
        print("분류 가능한 데이터가 없습니다.")
        return None
    result = {}
    for cat in df_usable['category'].unique():
        all_words = []
        df_usable[df_usable['category'] == cat]['matched_keywords'].apply(all_words.extend)
        result[cat] = Counter(all_words).most_common(15)
    return result

# 맥 한글 폰트
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

# --- 키워드 분석 결과 출력 ---
keyword_results = analyze_keywords_by_category(df)
if keyword_results:
    print("### 카테고리별 상위 키워드 ###")
    for cat, kws in keyword_results.items():
        print(f"{cat}: {[f'{w}({c})' for w, c in kws]}")

font_path = '/Library/Fonts/AppleGothic.ttf'

# 1) 카테고리별 데이터 수 막대그래프
plt.figure(figsize=(10,6), dpi=100)
category_counts = df['category'].value_counts()
bar_plot = category_counts.plot(kind='bar',
                                color='cornflowerblue', 
                                width=0.7) 
plt.title('카테고리별 데이터 수', fontsize=16, fontweight='bold')
plt.xlabel('카테고리', fontsize=12)
plt.ylabel('갯수', fontsize=12)
plt.tight_layout()
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10) 
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.tight_layout()
for i, v in enumerate(df['category'].value_counts()):
    plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
plt.show()

# 2) 카테고리별 상위 키워드 추출
def get_keyword_results(df_input):
    df_use = df_input[df_input['category'] != "분류불가"].copy()
    result = {}
    for cat in df_use['category'].unique():
        all_kw = []
        # matched_keywords 컬럼의 각 리스트들을 하나의 리스트로 합침
        for kws_list in df_use[df_use['category'] == cat]['matched_keywords']:
            all_kw.extend(kws_list)
        if all_kw: # 키워드가 있는 경우에만 Counter 실행
            result[cat] = Counter(all_kw).most_common(15)
    return result

keyword_results = get_keyword_results(df)

# 3) 전체 카테고리 통합 워드클라우드

all_kws = []
for kws_list in df['matched_keywords']:
    all_kws.extend(kws_list)

overall_freq = Counter(all_kws)
wc_all = WordCloud(
    font_path=font_path,
    width=800, height=600,
    background_color='white',
    max_font_size=180, colormap='viridis', random_state=42, prefer_horizontal=0.95
).generate_from_frequencies(overall_freq)

plt.figure(figsize=(12,9), dpi=100)
plt.imshow(wc_all, interpolation='bilinear')
plt.title('전체 카테고리 워드클라우드', fontsize=18, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()