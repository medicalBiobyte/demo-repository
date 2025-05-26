import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# ✅ 1. 환경변수 로딩
print("🔐 환경변수 로딩 중...")
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print("✅ OPENAI_API_KEY 로딩 완료")

# ✅ 2. CSV 경로
fnclty_path = "csv_data/fnclty_materials_complete.csv"
drug_path = "csv_data/drug_raw.csv"
healthfood_claims_path = "csv_data/healthfood_claims_final10.csv"  # 새 CSV 파일 경로 추가

# ✅ 3. CSV 로딩
print(f"📄 CSV 파일 로딩 중: {fnclty_path}")
df_fnclty = pd.read_csv(fnclty_path)
print(f"➡️ {len(df_fnclty)}개의 기능성 성분 로딩 완료")

print(f"📄 CSV 파일 로딩 중: {drug_path}")
df_drug = pd.read_csv(drug_path)
print(f"➡️ {len(df_drug)}개의 의약품 데이터 로딩 완료")

print(f"📄 CSV 파일 로딩 중: {healthfood_claims_path}") # 새 CSV 파일 로딩
try:
    df_healthfood_claims = pd.read_csv(healthfood_claims_path)
    print(f"➡️ {len(df_healthfood_claims)}개의 건강기능식품 클레임 데이터 로딩 완료")
except FileNotFoundError:
    print(f"❌ 오류: {healthfood_claims_path} 파일을 찾을 수 없습니다. 이 파일에 대한 처리를 건너뜁니다.")
    df_healthfood_claims = pd.DataFrame() # 빈 데이터프레임으로 초기화하여 이후 로직에서 오류 방지


# ✅ 4. 문서로 변환 함수
def fnclty_to_doc(row):
    text = f"{str(row.get('APLC_RAWMTRL_NM', ''))} - {str(row.get('FNCLTY_CN', ''))}"
    metadata = {
        "source": "fnclty",
        "material": str(row.get("APLC_RAWMTRL_NM", "")),
        "functionality": str(row.get("FNCLTY_CN", "")),
        "intake": str(row.get("DAY_INTK_CN", "")),
        "caution": str(row.get("IFTKN_ATNT_MATR_CN", "")),
    }
    return Document(page_content=text, metadata=metadata)


def drug_to_doc(row):
    text = f"{str(row.get('itemName', ''))} - {str(row.get('efcyQesitm', ''))}"
    metadata = {
        "source": "drug",
        "product": str(row.get("itemName", "")),
        "manufacturer": str(row.get("entpName", "")),
        "efficacy": str(row.get("efcyQesitm", "")),
        "usage": str(row.get("useMethodQesitm", "")),
        "caution": str(row.get("atpnQesitm", "")),
    }
    return Document(page_content=text, metadata=metadata)

# healthfood_claims_final10.csv를 위한 문서 변환 함수
def healthfood_claims_to_doc(row):
    product_name = str(row.get('제품명', '')).strip()
    functionality_content = str(row.get('기능성 내용', '')).strip()   
    daily_intake = str(row.get('일일섭취량', '')).strip()
    precautions = str(row.get('섭취 시 주의사항', '')).strip()
    
    # 👇 "밀크씨슬" 관련 행인지 확인하고 내부 데이터 출력 (디버깅용)
    if "밀크씨슬" in product_name or "밀크씨슬" in functionality_content:
        print(f"\n[healthfood_claims_to_doc 디버깅] \"밀크씨슬\" 포함 행 발견:")
        print(f"  > 원본 행 데이터 (일부): 제품명='{product_name}', 기능성내용='{functionality_content}'")
    if not product_name or not functionality_content:
        if "밀크씨슬" in product_name or "밀크씨슬" in functionality_content: # 디버깅
            print(f"  > ⚠️ 필수 정보 부족으로 이 문서는 생성되지 않습니다.")
        return None
    
    if not product_name or not functionality_content: # 제품명과 기능성 내용이 없으면 유효하지 않은 문서로 간주
        return None
        
    # RAG 검색 시 핵심이 될 page_content 구성
    text_parts = [
        f"건강기능식품 제품명: {product_name}",
        f"주요 기능성 내용: {functionality_content}"
    ]
    if daily_intake:
        text_parts.append(f"일일 섭취량: {daily_intake}")
    if precautions:
        text_parts.append(f"섭취 시 주의사항: {precautions}")
    
    text = ". ".join(text_parts) + "." # 각 정보를 문장으로 연결

    metadata = {
        "source": "healthfood_claims_final10", # 출처 명시
        "product_name": product_name,
        "functionality_content": functionality_content, # 메타데이터에도 주요 정보 포함
        "daily_intake": daily_intake if daily_intake else "정보 없음",
        "precautions": precautions if precautions else "정보 없음",
        "original_filename": str(row.get('파일명', '')).strip() # 원본 CSV의 '파일명' 컬럼 추가
    }
    # 👇 생성된 Document 객체의 page_content 와 metadata 출력 (디버깅용)
    if "밀크씨슬" in product_name or "밀크씨슬" in functionality_content:
        print(f"  > 생성될 page_content: {text[:200]}...") # 앞 200자만 출력
        print(f"  > 생성될 metadata: {metadata}")

    return Document(page_content=text, metadata=metadata)

# ✅ 5. 문서 리스트 생성
print("📦 문서 리스트 생성 중...")
docs_fnclty = [fnclty_to_doc(row) for _, row in df_fnclty.iterrows()]
docs_drug = [drug_to_doc(row) for _, row in df_drug.iterrows()]
docs_healthfood_claims = [healthfood_claims_to_doc(row) for _, row in df_healthfood_claims.iterrows()]
all_docs = docs_fnclty + docs_drug + docs_healthfood_claims 
print(f"✅ 총 {len(all_docs)}개의 문서 변환 완료")

# ✅ 6. 임베딩 및 벡터 DB 초기화
print("🔧 벡터 DB 초기화 중...")
embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
vector_db = Chroma(
    persist_directory="./chroma_db",
    collection_name="health_collection",
    embedding_function=embedding,
)
print("✅ 벡터 DB 초기화 완료")

# ✅ 7. 문서 삽입 및 저장
print(f"📤 총 {len(all_docs)}개의 문서 임베딩 및 벡터스토어에 저장 준비 중...")
batch_size = 1000  # 한 번에 처리할 문서 수
num_docs = len(all_docs)

print(f"ℹ️ 문서를 {batch_size}개씩 나누어 저장합니다.")
for i in range(0, num_docs, batch_size):
    batch_docs = all_docs[i:i + batch_size]
    print(f"  ➡️ 배치 {i//batch_size + 1}: {len(batch_docs)}개 문서 처리 중 ({i + len(batch_docs)}/{num_docs})...")
    try:
        vector_db.add_documents(documents=batch_docs) # LangChain Chroma는 documents 매개변수명을 사용합니다.
    except Exception as e:
        print(f"  ❌ 오류 발생! 배치 {i//batch_size + 1} 처리 중 문제 발생: {e}")
        print(f"    해당 배치의 첫 번째 문서 (일부): {str(batch_docs[0])[:200] if batch_docs else '없음'}")
        # 필요하다면 여기서 반복을 중단하거나, 오류가 발생한 배치를 건너뛰는 로직 추가 가능
        break # 오류 발생 시 일단 중단
else: # for 루프가 break 없이 정상적으로 완료되었을 때 실행
    print("🎉 모든 문서 배치 저장 완료! ✅")
    try:
        # 최종 문서 수 확인 (컬렉션이 실제로 존재하는지 확인 후 접근)
        if vector_db._collection is not None:
                print(f"✨ ChromaDB 컬렉션 '{vector_db._collection.name}'에 현재 {vector_db._collection.count()}개의 문서가 저장되어 있습니다.")
        else:
            print("⚠️ ChromaDB 컬렉션 정보를 가져올 수 없습니다.")
    except Exception as e:
        print(f"⚠️ ChromaDB 문서 수 확인 중 오류: {e}")