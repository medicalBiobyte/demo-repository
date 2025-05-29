import pandas as pd
from konlpy.tag import Okt
from collections import Counter
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import sys
import os  # OS 모듈 추가

# --- 파일 경로 설정 ---
fnclty_path = "csv_data/fnclty_materials_complete.csv"
drug_path = "csv_data/drug_raw.csv"
healthfood_claims_path = "csv_data/healthfood_claims_final10.csv"

# --- Windows 환경 한글 폰트 설정 ---
# 맑은 고딕 또는 다른 한글 지원 폰트 경로를 찾아서 설정합니다.
# 일반적으로 'C:/Windows/Fonts/malgun.ttf'에 있습니다.
if sys.platform == "win32":
    plt.rc("font", family="Malgun Gothic")  # 맑은 고딕으로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 폰트 경로
    # 만약 맑은 고딕이 없거나 다른 폰트를 사용하고 싶다면, 해당 폰트 파일의 경로를 지정해주세요.
    # 예: 'C:/Windows/Fonts/NanumGothic.ttf' (나눔고딕을 설치했을 경우)
else:  # Mac 또는 Linux 환경
    plt.rc("font", family="AppleGothic")
    font_path = "/Library/Fonts/AppleGothic.ttf"  # 맥 한글 폰트 경로

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 폰트 깨짐 방지

# --- 분석 대상 파일 및 컬럼 매핑 ---
file_configs = [
    (healthfood_claims_path, "기능성 내용", "healthfood_claims"),
    (drug_path, "efcyQesitm", "drug_raw"),
    (fnclty_path, "FNCLTY_CN", "fnclty_materials"),
]


# --- 형태소 분석기 및 불용어 설정 ---
def run_pipeline(df, label):
    okt = Okt()
    stopwords = [
        "에",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "들",
        "좀",
        "잘",
        "걍",
        "과",
        "도",
        "으로",
        "자",
        "와",
        "한",
        "하다",
        "되다",
        "있다",
        "없다",
        "이다",
        "좋다",
        "많다",
        "그리고",
        "하지만",
        "그러나",
        "그래서",
        "또는",
        "및",
        "등",
        "수",
        "것",
        "줄",
        "알",
        "더",
        "그",
        "저",
        "이런",
        "저런",
        "그런",
        "위해",
        "때문",
        "통해",
        "대한",
        "관련",
        "도움",
        "기능성",
        "완화",
        "개선",
        "예방",
    ]

    # --- 텍스트 전처리 함수 ---
    def preprocess_text(text):
        if pd.isna(text) or not text:
            return []
        text_clean = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\s]", " ", str(text))
        nouns = okt.nouns(text_clean)
        return [w for w in nouns if w not in stopwords and len(w) > 1]

    # --- 카테고리 키워드 사전 (MFDS 기능성 영역 기반) ---
    # (카테고리 키워드 사전은 변경 없음)
    category_keywords = {
        "신경계": [
            "인지기능",
            "기억력",
            "집중력",
            "주의력",
            "학습능력",
            "스트레스 완화",
            "진정",
            "불안 개선",
            "기분 조절",
            "피로회복",
            "뇌피로",
            "멜라토닌",
            "가바",
            "수면개선",
            "수면의질향상",
            "불면증완화",
            "신경통완화",
            "두통완화",
            "어지럼증",
            "근육경련",
        ],
        "소화/대사계": [
            "위 건강",
            "위장운동",
            "소화불량",
            "숙취해소",
            "속쓰림",
            "위통",
            "체함",
            "구역",
            "위산과다",
            "위궤양",
            "제산",
            "위염",
            "위부팽만감",
            "구토억제",
            "장 건강",
            "배변활동",
            "변비",
            "설사 개선",
            "장염",
            "과민성대장증후군",
            "치질완화",
            "항문질환",
            "프로바이오틱스",
            "프리바이오틱스",
            "유산균",
            "장내환경개선",
            "유익균증식",
            "유해균억제",
            "체지방 감소",
            "대사조절",
            "포만감",
            "식욕조절",
            "식욕부진개선",
            "비만",
            "칼슘흡수",
            "칼슘대사",
            "비타민D",
            "간 건강",
            "간기능개선",
            "간보호",
            "간염보조치료",
            "간해독",
            "지방간",
            "간경변",
            "간질환",
        ],
        "생식&비뇨계": [
            "전립선 건강",
            "전립선비대증",
            "배뇨 기능",
            "요로 건강",
            "방광염",
            "요로감염 예방",
            "신장기능",
            "요로결석보조",
            "성기능 개선",
            "남성호르몬 균형",
            "발기부전",
            "생리불순",
            "질 건강",
            "여성호르몬 균형",
            "생리통완화",
            "월경전증후군",
            "칸디다질염",
            "세균성질염",
            "갱년기 증상 완화",
            "피임",
        ],
        "신체방어·면역계": [
            "면역력 강화",
            "면역세포 증강",
            "NK세포 활성화",
            "알레르기반응조절",
            "코과민반응개선",
            "항산화",
            "염증 완화",
            "항바이러스",
            "항균 작용",
            "인삼",
            "홍삼",
            "베타글루칸",
            "프로폴리스",
            "아연",
            "셀레늄",
            "호흡기 건강",
            "기관지 건강",
            "폐 건강",
            "코 건강",
            "목 건강",
            "기침",
            "가래",
            "천식",
            "기관지염",
            "비염",
            "코막힘",
            "콧물",
            "재채기",
            "인후염",
            "편도염",
            "상기도감염",
            "알레르기성 비염",
        ],
        "감각계": [
            "눈 건강",
            "시력 보호",
            "황반변성 예방",
            "안구건조 완화",
            "눈의피로",
            "결막염치료",
            "각막보호",
            "콘택트렌즈관리",
            "인공눈물",
            "다래끼",
            "녹내장",
            "고안압",
            "청각 보호",
            "이명 완화",
            "귀건강",
            "이명",
            "이명증",
            "치아 건강",
            "잇몸 강화",
            "구강 위생",
            "치주질환",
            "치은염",
            "구내염",
            "설염",
            "입냄새제거",
            "피부 건강",
            "보습",
            "탄력",
            "자외선 차단",
            "동상",
            "습진",
            "피부염",
            "아토피",
            "화상",
            "상처치료",
            "피부궤양",
            "피부재생",
            "가려움해소",
            "두드러기완화",
            "여드름치료",
            "뾰루지",
            "무좀치료",
            "백선",
            "건선",
            "벌레물림",
            "피부소독",
            "살균",
            "상처소독",
            "땀띠",
            "발진",
            "지루성피부염",
            "비듬",
            "티눈",
            "굳은살",
            "사마귀",
            "다한증",
            "기미",
            "눈의 세정",
            "눈의 불쾌감",
        ],
        "심혈관계": [
            "혈압 조절",
            "고혈압 완화",
            "혈관 탄력",
            "혈중 중성지방 개선",
            "콜레스테롤개선",
            "LDL",
            "HDL 균형",
            "혈행 개선",
            "말초혈관 순환",
            "혈액순환",
            "혈소판응집억제",
            "동맥경화 예방",
            "혈색소 관리",
            "빈혈예방",
            "철분보충",
            "나토키나제",
            "홍국",
            "코엔자임Q10",
            "심부전보조치료",
        ],
        "내분비계": [
            "혈당 조절",
            "인슐린 저항성",
            "식후혈당 안정화",
            "당뇨병보조",
            "갑상선 대사",
            "갑상선 기능 개선",
            "갱년기 증상 완화",
            "월경전증후군 개선",
            "생리통 완화",
            "블랙코호시",
            "보스웰리아",
        ],
        "근육계": [
            "뼈 건강",
            "관절 건강",
            "연골 보호",
            "관절염 완화",
            "타박상",
            "삠",
            "근력 강화",
            "근지구력",
            "근육통 완화",
            "어꺠결림",
            "류마티스 통증",
            "운동수행능력",
            "회복력",
            "근육피로 회복",
            "MSM",
            "글루코사민",
            "콘드로이틴",
            "콜라겐",
            "비타민C",
            "어린이키성장",
            "성장기영양",
            "근육통증",
        ],
    }

    # --- 카테고리 분류 함수 ---
    def assign_category_with_hits(tokens):
        if not tokens:
            return "기타", []
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
            return "기타", []
        best_cat = Counter(hits).most_common(1)[0][0]
        filtered_matches = [
            kw for kw in matched_kws if any(c == best_cat for c in hits)
        ]
        return best_cat, list(set(filtered_matches))

    # --- 분류 및 매칭된 키워드 컬럼 추가 ---
    df[["category", "matched_keywords"]] = (
        df["기능성 내용"]
        .apply(preprocess_text)
        .apply(lambda tokens: pd.Series(assign_category_with_hits(tokens)))
    )

    # --- 결과 출력 ---
    print("### 카테고리 분류 결과 (상위 5개) ###")
    print(df[["기능성 내용", "matched_keywords", "category"]].head(), "\n")
    print("### 카테고리별 데이터 수 ###")
    print(df["category"].value_counts(), "\n")

    # --- 카테고리별 키워드 빈도 분석 함수 ---
    def analyze_keywords_by_category(df_input):
        if (
            "category" not in df_input.columns
            or "matched_keywords" not in df_input.columns
        ):
            print("오류: 'category' 또는 'matched_keywords' 컬럼이 없습니다.")
            return None
        df_usable = df_input[df_input["category"] != "기타"]
        if df_usable.empty:
            print("분류 가능한 데이터가 없습니다.")
            return None
        result = {}
        for cat in df_usable["category"].unique():
            all_words = []
            df_usable[df_usable["category"] == cat]["matched_keywords"].apply(
                all_words.extend
            )
            result[cat] = Counter(all_words).most_common(15)
        return result

    # --- 키워드 분석 결과 출력 ---
    keyword_results = analyze_keywords_by_category(df)
    if keyword_results:
        print("### 카테고리별 상위 키워드 ###")
        for cat, kws in keyword_results.items():
            print(f"{cat}: {[f'{w}({c})' for w, c in kws]}")

    # 1) 카테고리별 데이터 수 막대그래프
    plt.figure(figsize=(12, 7), dpi=100)
    category_counts = df["category"].value_counts()
    bar_plot = category_counts.plot(kind="bar", color="lightskyblue", width=0.9)
    plt.title("카테고리별 데이터 수", fontsize=18, fontweight="bold")
    plt.xlabel("카테고리", fontsize=14, fontweight="bold")
    plt.ylabel("갯수", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.xticks(rotation=0, ha="center", fontsize=11)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    for i, v in enumerate(df["category"].value_counts()):
        plt.text(i, v + 0.2, str(v), ha="center", va="bottom")
    plt.show()

    # 2) 카테고리별 상위 키워드 추출
    def get_keyword_results(df_input):
        df_use = df_input[df_input["category"] != "기타"].copy()
        result = {}
        for cat in df_use["category"].unique():
            all_kw = []
            # matched_keywords 컬럼의 각 리스트들을 하나의 리스트로 합침
            for kws_list in df_use[df_use["category"] == cat]["matched_keywords"]:
                all_kw.extend(kws_list)
            if all_kw:  # 키워드가 있는 경우에만 Counter 실행
                result[cat] = Counter(all_kw).most_common(15)
        return result

    keyword_results = get_keyword_results(df)

    # 3) 전체 카테고리 통합 워드클라우드
    all_kws = []
    for kws_list in df["matched_keywords"]:
        all_kws.extend(kws_list)

    overall_freq = Counter(all_kws)
    # WordCloud에 나타나지 않게 할 일반 용어(stopwords 추가)
    custom_stopwords = STOPWORDS.union({"완화", "예방", "개선"})
    wc_all = WordCloud(
        font_path=font_path,  # 설정된 폰트 경로 사용
        width=800,
        height=600,
        background_color="white",
        stopwords=custom_stopwords,
        max_font_size=180,
        colormap="PuBuGn",
        random_state=42,
        prefer_horizontal=0.95,
    ).generate_from_frequencies(overall_freq)

    plt.figure(figsize=(12, 9), dpi=100)
    plt.imshow(wc_all, interpolation="bilinear")
    plt.title("워드클라우드", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    print(f"\n### {label} 데이터 처리 완료 ###\n")
    return df


# --- 실행부 -------------------------------------------------------------
if __name__ == "__main__":
    # 콘솔 출력 인코딩을 UTF-8로 설정 (Windows 환경에서 한글 깨짐 방지)
    # Jupyter Notebook/IPython 환경에서는 필요 없을 수 있습니다.
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout = open(
                sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1
            )
            sys.stderr = open(
                sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1
            )
            print("콘솔 인코딩이 UTF-8로 설정되었습니다.")
        except Exception as e:
            print(f"경고: 콘솔 인코딩을 UTF-8로 설정하는 데 실패했습니다. ({e})")
            print("명령 프롬프트에서 'chcp 65001'을 먼저 실행해 보세요.")

    for path, eff_col, label in file_configs:
        # 파일 존재 여부 확인 및 경로 출력
        if not os.path.exists(path):
            print(f"[오류] 파일이 존재하지 않습니다: {path}")
            continue

        print(f"\n--- {label} 데이터 분석 시작: {path} ---")
        try:
            # CSV 파일 읽을 때 'encoding' 인자를 'utf-8'로 명시
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            print(
                f"[경고] {path} 파일이 UTF-8로 디코딩되지 않았습니다. cp949로 재시도합니다."
            )
            try:
                df = pd.read_csv(path, encoding="cp949")
            except Exception as e:
                print(
                    f"[오류] {path} 파일을 읽는 데 실패했습니다. ({e}) 해당 파일의 인코딩을 확인해주세요."
                )
                continue
        except Exception as e:
            print(f"[오류] {path} 파일을 읽는 데 실패했습니다. ({e})")
            continue

        if eff_col not in df.columns:
            print(f"[경고] '{eff_col}' 컬럼이 {label} 파일에 없습니다.")
            continue
        # 공통 컬럼명으로 통일
        df["기능성 내용"] = df[eff_col]
        df_processed = run_pipeline(df, label)
        # unclassified_items 저장 로직 (필요시 주석 해제하여 사용)
        # if label != 'healthfood_claims': # 기타 키워드 찾는용
        #     unclassified_items = df_processed[df_processed['category'] == '기타']['기능성 내용']
        #     if not unclassified_items.empty:
        #         output_filename = f"unclassified_{label}_items.txt"
        #         try:
        #             unclassified_items.to_csv(output_filename, index=False, header=False, encoding='utf-8')
        #             print(f"\n'{label}' 데이터의 '기타' 항목 ({len(unclassified_items)}개)이 '{output_filename}' 파일로 저장되었습니다.")
        #             print(f"'{output_filename}' 파일을 열어 내용을 확인하고 키워드 사전을 업데이트해 보세요!")
        #         except Exception as e:
        #             print(f"파일 저장 중 오류 발생: {e}")
        #     else:
        #         print(f"\n'{label}' 데이터에 '기타' 항목이 없습니다.")
