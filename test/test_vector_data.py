from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from collections import Counter

# ✅ 벡터 임베딩 모델
embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# ✅ Chroma VectorDB 로드
vector_db = Chroma(
    persist_directory="./chroma_db",
    collection_name="health_collection",
    embedding_function=embedding,
)

# ✅ 내부 collection에서 모든 metadata 추출
collection = vector_db._collection
all_metadata = collection.get(include=["metadatas"], limit=10000)["metadatas"]

# ✅ source 필드만 수집
source_list = [m.get("source", "unknown") for m in all_metadata if m]

# ✅ source별 문서 수 분석
source_counts = Counter(source_list)

# ✅ 결과 출력
for source, count in source_counts.items():
    print(f"📁 source: {source} — 총 {count}건")
