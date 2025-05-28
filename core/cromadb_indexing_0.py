import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


# 문서 변환 함수들
def fnclty_to_doc(row):
    return Document(
        page_content=f"{row.get('APLC_RAWMTRL_NM', '')} - {row.get('FNCLTY_CN', '')}",
        metadata={
            "source": "fnclty",
            "material": row.get("APLC_RAWMTRL_NM", ""),
            "functionality": row.get("FNCLTY_CN", ""),
            "intake": row.get("DAY_INTK_CN", ""),
            "caution": row.get("IFTKN_ATNT_MATR_CN", ""),
        },
    )


def drug_to_doc(row):
    return Document(
        page_content=f"{row.get('itemName', '')} - {row.get('efcyQesitm', '')}",
        metadata={
            "source": "drug",
            "product": row.get("itemName", ""),
            "manufacturer": row.get("entpName", ""),
            "efficacy": row.get("efcyQesitm", ""),
            "usage": row.get("useMethodQesitm", ""),
            "caution": row.get("atpnQesitm", ""),
        },
    )


def healthfood_claims_to_doc(row):
    if not row.get("제품명") or not row.get("기능성 내용"):
        return None
    text = (
        f"건강기능식품 제품명: {row['제품명']}. 주요 기능성 내용: {row['기능성 내용']}."
    )
    if row.get("일일섭취량"):
        text += f" 일일 섭취량: {row['일일섭취량']}."
    if row.get("섭취 시 주의사항"):
        text += f" 섭취 시 주의사항: {row['섭취 시 주의사항']}."
    return Document(
        page_content=text,
        metadata={
            "source": "healthfood_claims_final10",
            "product_name": row.get("제품명", ""),
            "functionality_content": row.get("기능성 내용", ""),
            "daily_intake": row.get("일일섭취량", "정보 없음"),
            "precautions": row.get("섭취 시 주의사항", "정보 없음"),
            "original_filename": row.get("파일명", ""),
        },
    )


# Main 실행부
if __name__ == "__main__":
    print("🔐 환경변수 로딩 중...")
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("✅ OPENAI_API_KEY 로딩 완료")

    fnclty_path = "csv_data/fnclty_materials_complete.csv"
    drug_path = "csv_data/drug_raw.csv"
    healthfood_claims_path = "csv_data/healthfood_claims_final10.csv"

    # CSV 로딩
    print(f"📄 CSV 파일 로딩 중: {fnclty_path}")
    df_fnclty = pd.read_csv(fnclty_path)
    print(f"➡️ {len(df_fnclty)}개의 기능성 성분 로딩 완료")

    print(f"📄 CSV 파일 로딩 중: {drug_path}")
    df_drug = pd.read_csv(drug_path)
    print(f"➡️ {len(df_drug)}개의 의약품 데이터 로딩 완료")

    print(f"📄 CSV 파일 로딩 중: {healthfood_claims_path}")
    try:
        df_healthfood_claims = pd.read_csv(healthfood_claims_path)
        print(f"➡️ {len(df_healthfood_claims)}개의 건강기능식품 클레임 데이터 로딩 완료")
    except FileNotFoundError:
        print(f"❌ 오류: {healthfood_claims_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
        df_healthfood_claims = pd.DataFrame()

    # 문서 변환
    print("📦 문서 리스트 생성 중...")
    docs_fnclty = [fnclty_to_doc(row) for _, row in df_fnclty.iterrows()]
    docs_drug = [drug_to_doc(row) for _, row in df_drug.iterrows()]
    docs_healthfood_claims = [
        doc
        for _, row in df_healthfood_claims.iterrows()
        if (doc := healthfood_claims_to_doc(row)) is not None
    ]
    all_docs = docs_fnclty + docs_drug + docs_healthfood_claims
    print(f"✅ 총 {len(all_docs)}개의 문서 변환 완료")

    # 벡터 DB 초기화
    print("🔧 벡터 DB 초기화 중...")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    vector_db = Chroma(
        persist_directory="./chroma_db",
        collection_name="health_collection",
        embedding_function=embedding,
    )
    print("✅ 벡터 DB 초기화 완료")

    # 배치 저장
    print(f"📤 총 {len(all_docs)}개 문서 저장 중...")
    batch_size = 1000
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        try:
            print(f"➡️ 저장 중: 문서 {i + 1} ~ {i + len(batch)}")
            vector_db.add_documents(batch)
        except Exception as e:
            print(f"❌ 배치 저장 실패 (index {i}): {e}")
            break
    else:
        print("🎉 모든 문서 저장 완료")

    # 저장된 문서 수 확인
    try:
        if vector_db._collection:
            print(f"✨ 저장된 문서 수: {vector_db._collection.count()}")
    except Exception as e:
        print(f"⚠️ 문서 수 확인 중 오류: {e}")
