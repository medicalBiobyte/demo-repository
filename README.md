# 💊 Health Product Claim Verification System

이 프로젝트는 건강기능식품 또는 의약외품 광고에서 주장하는 효능이 실제로 성분 기반 공공 데이터와 일치하는지 검증하는 시스템입니다. 이미지 인식부터 웹 검색, 벡터 검색, LLM 기반 판단까지 전체 파이프라인을 구성합니다.

---

## 🔧 구성 파일 및 설명

### 1. `cromadb_indexing_0.py`  
기능성 원료와 의약품 데이터를 벡터 임베딩하여 Chroma 벡터스토어에 저장합니다.
- 입력 데이터: `csv_data/fnclty_materials_complete.csv`, `drug_raw.csv`
- 출력 위치: `./chroma_db/`
- 사용 모델: `text-embedding-3-small`

---

### 2. `text_extract_1.py`  
제품 광고 이미지에서 텍스트를 추출하고, 제품명과 효능 주장을 LLM을 통해 구조화합니다.
- 입력 이미지 디렉토리: `img/`
- 출력 JSON: `IMG2TEXT_data/result_all.json`
- 사용 모델: `Gemini (image_llm)`
- 프롬프트: `IMG2TEXT_PROMPT`

---

### 3. `web_search_2.py`  
제품명을 기반으로 웹 검색을 실행하고, 검색 결과에서 성분 및 효능 정보를 정리합니다.
- 입력: `IMG2TEXT_data/result_all.json`
- 출력: `TEXT2SEARCH_data/enriched_{제품명}.json`
- 사용 모델: `Tavily (검색) + GPT (정리)`
- 프롬프트: `WEB2INGREDIENT_PROMPT`

---

### 4. `claim_check_3.py`  
사용자의 질문과 제품의 성분 정보를 비교하여, 효능 주장이 일치하는지 판단합니다.
- 입력: `TEXT2SEARCH_data/enriched_*.json`, 사용자 질문
- 출력: `DECISION_data/enriched_{제품명}.json`
- 사용 모델: GPT
- 프롬프트: `QUERY2KEYWORD_PROMPT`

---

### 5. `answer_user_4.py`  
최종 판단 결과 JSON을 사람이 읽기 쉬운 자연어 설명으로 변환합니다.
- 입력: `DECISION_data/`
- 출력: 콘솔에 텍스트 출력 또는 API 응답용 사용 가능

---

### 6. `prompt.py`  
LLM에 전달할 프롬프트 템플릿을 모아둔 파일입니다.
- `IMG2TEXT_PROMPT`: 이미지에서 제품명/효능 주장 추출
- `WEB2INGREDIENT_PROMPT`: 웹 검색 결과에서 성분 추출
- `QUERY2KEYWORD_PROMPT`: 질문에서 핵심 효능 키워드 추출

---

### 7. `config.py`  
전체 시스템에서 사용하는 API 키, LLM 설정, 검색 클라이언트 등을 구성합니다.
- 사용 LLM: `gpt-4o-mini`, `Gemini`, `Tavily`
- 벡터스토어: Chroma + OpenAI Embedding

---

## 🔄 전체 파이프라인 흐름

1. **이미지 입력**
   → `text_extract_1.py`: 제품명 및 광고 문구 추출  
2. **웹 검색**
   → `web_search_2.py`: 성분 정보 및 효능 수집  
3. **사용자 질문과 비교**
   → `claim_check_3.py`: 공공 DB와 비교하여 주장 검증  
4. **자연어 응답**
   → `answer_user_4.py`: 사람 친화적인 설명 출력

---

## ✅ 예시 실행 순서

```bash
python cromadb_indexing_0.py      # 벡터 DB 구성
python text_extract_1.py          # 이미지 정보 추출
python web_search_2.py            # 성분 웹 검색
python claim_check_3.py           # 질문 기반 검증
python answer_user_4.py           # 자연어 응답 출력
```

---

## 📂 디렉토리 구조

```
csv_data/
    fnclty_materials_complete.csv
    drug_raw.csv

img/
    광고 이미지들

IMG2TEXT_data/
    result_all.json

TEXT2SEARCH_data/
    enriched_{제품명}.json

DECISION_data/
    enriched_{제품명}.json

chroma_db/
    벡터 임베딩 데이터
```

---

## 💬 문의 및 기여

- 작성자: 영환  
- 사용 기술: Python, LangChain, Chroma, GPT-4o, Gemini, Tavily  
- 기여 및 문의: 
