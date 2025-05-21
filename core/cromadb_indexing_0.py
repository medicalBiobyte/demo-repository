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

# ✅ 3. CSV 로딩
print(f"📄 CSV 파일 로딩 중: {fnclty_path}")
df_fnclty = pd.read_csv(fnclty_path)
print(f"➡️ {len(df_fnclty)}개의 기능성 성분 로딩 완료")

print(f"📄 CSV 파일 로딩 중: {drug_path}")
df_drug = pd.read_csv(drug_path)
print(f"➡️ {len(df_drug)}개의 의약품 데이터 로딩 완료")


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


# ✅ 5. 문서 리스트 생성
print("📦 문서 리스트 생성 중...")
docs_fnclty = [fnclty_to_doc(row) for _, row in df_fnclty.iterrows()]
docs_drug = [drug_to_doc(row) for _, row in df_drug.iterrows()]
all_docs = docs_fnclty + docs_drug
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
print("📤 문서 임베딩 및 벡터스토어에 저장 중...")
vector_db.add_documents(all_docs)
# vector_db.persist()
print("🎉 벡터스토어 저장 완료! ✅")
